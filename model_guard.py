"""
Provenance pinning and pre-activation validation for model artifacts.

WHY THIS EXISTS
---------------
Every model in this repo enters through `from_pretrained(...)` /
`snapshot_download(...)` reading a cache nobody verifies. On the Modal side that
is a PERSISTENT, MUTABLE Volume (`turboquant-hf-cache`, `llmlingua2-hf-cache`,
`hf-cache`): populate-once-then-read, committed back from inside the GPU
container, and it outlives every image rebuild, so whatever sits in it at load
time is what gets loaded. On the `turboquant_kv/` side it is the plain HF cache
on a shared benchmark box. Before this module none of the load sites pinned a
revision, forced safetensors, or checked a digest - the cache was an unverified
trust boundary in both places.

That is not hypothetical here. `BAAI/bge-small-en-v1.5` (the SmallEmbedder in
`two_stage_compressor`) publishes BOTH `model.safetensors` and
`pytorch_model.bin`. A `.bin` checkpoint is a zipped Python pickle, and
unpickling is arbitrary code execution: the payload runs during `load`, before
any shape or dtype is ever inspected. Whether the installed `transformers`
happens to pass `weights_only=True` is not something this repo controls - the
Modal images pip-install `transformers` unpinned. So we take the file out of the
decision instead of trusting the loader.

Worse, one dependency opts IN for us. llmlingua 0.2.2 - the version
`llmlingua2_modal.py` pins - defaults `trust_remote_code` to True and forwards
it into `AutoConfig`, `AutoTokenizer` and the model class, so any `*.py` sitting
in that mutable volume would be imported and executed at load time. See
`llmlingua_model_config` below for the library source that does it.

WHAT THIS MODULE ENFORCES
-------------------------
  1. provenance - a pinned commit SHA per model id (`REVISIONS`), so a change to
     the hub's `main` branch cannot silently change the weights.
  2. no code    - `trust_remote_code=False`, and `verify_snapshot` refuses a
     snapshot that contains any `*.py`.
  3. no pickle  - `use_safetensors=True`. If a pickle must be read at all it
     goes through `scan_pickle`, an opcode-level allowlist that runs BEFORE any
     unpickling, and only then through `torch.load(..., weights_only=True)`.
  4. validation BEFORE activation - digests and byte sizes against a manifest,
     then tensor key order, shapes, dtypes and finiteness.
  5. rollback   - `KnownGood` keeps the last manifest that passed, so a rejected
     candidate leaves the previously-good revision in place and available.

Only the stdlib is used on the validation path (`pickletools`, `hashlib`,
`json`, `zipfile`). `torch` and `safetensors` are imported lazily and only when
tensors are actually read, so this module imports - and its tests run - on a
CPU-only box with neither installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickletools
import warnings
import zipfile
from typing import Dict, Mapping, Optional, Sequence


class ArtifactRejected(Exception):
    """An artifact failed verification and must not be activated."""


class VerifiedSnapshot(str):
    """A local snapshot path whose bytes passed `verify_hf_snapshot`."""

    def __new__(cls, path: str, model_id: str, manifest: Mapping):
        value = super().__new__(cls, path)
        value.model_id = model_id
        value.manifest = dict(manifest)
        return value

    def __reduce__(self):
        """Keep provenance attached when libraries deepcopy their arguments."""
        return type(self), (str(self), self.model_id, self.manifest)


_ACTIVE_MODELS: Dict[tuple, object] = {}


# --------------------------------------------------------------------------- #
# 1. Provenance: pinned revisions                                             #
# --------------------------------------------------------------------------- #
# Commit SHAs read from https://huggingface.co/api/models/<id> on 2026-09-05.
# Pinning is the whole point: bump these deliberately, never automatically.
REVISIONS: Dict[str, str] = {
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank":
        "ebaba9b0e874dadd3003ffcff828e4397e568089",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen/Qwen2.5-14B-Instruct": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}

def pinned_revision(model_id: str) -> str:
    """Return the pinned commit SHA for `model_id`, or refuse.

    Fail-closed: an unknown model id is an unreviewed artifact, so it is
    rejected rather than resolved to whatever `main` points at today.
    """
    rev = REVISIONS.get(model_id)
    if not rev:
        raise ArtifactRejected(
            f"no pinned revision for {model_id!r}; add its commit SHA to "
            "model_guard.REVISIONS after reviewing the artifact"
        )
    return rev


def guarded_kwargs(model_id: str, **kwargs) -> dict:
    """Return `kwargs` with the non-negotiable load flags forced on.

    Refuses rather than silently downgrades if a caller tries to opt out of
    either safety flag - an override would have to be argued for in code review,
    not passed at a call site.
    """
    if kwargs.get("trust_remote_code"):
        raise ArtifactRejected(
            f"trust_remote_code=True on {model_id!r} executes repo-supplied "
            "Python at load time; refused"
        )
    if kwargs.get("use_safetensors") is False:
        raise ArtifactRejected(
            f"use_safetensors=False on {model_id!r} allows the pickled .bin "
            "checkpoint path; refused"
        )
    pinned = pinned_revision(model_id)
    requested = kwargs.get("revision")
    if requested is not None and requested != pinned:
        raise ArtifactRejected(
            f"{model_id!r} must use pinned revision {pinned}; "
            f"caller requested {requested!r}"
        )
    kwargs["revision"] = pinned
    kwargs["use_safetensors"] = True
    kwargs["trust_remote_code"] = False
    return kwargs


def guarded_from_pretrained(loader, model_id: str, **kwargs):
    """`loader.from_pretrained(model_id, ...)` with the guard flags applied.

    `loader` is an Auto* class (AutoTokenizer, AutoModelForCausalLM, ...).
    Tokenizers have no safetensors weights, so `use_safetensors` is dropped for
    them; the revision pin and the remote-code refusal still apply.
    """
    kw = guarded_kwargs(model_id, **kwargs)
    is_tokenizer = "Tokenizer" in getattr(loader, "__name__", "")
    download_kwargs = {
        key: kw.pop(key)
        for key in ("cache_dir", "token", "local_files_only", "force_download")
        if key in kw
    }
    activation_key = (
        model_id,
        getattr(loader, "__module__", ""),
        getattr(loader, "__qualname__", getattr(loader, "__name__", "")),
    )
    previous = None if is_tokenizer else _ACTIVE_MODELS.get(activation_key)
    try:
        snapshot_dir = verified_snapshot_download(
            model_id, include_weights=not is_tokenizer, **download_kwargs
        )
        kw.pop("revision", None)
        kw["local_files_only"] = True
        if is_tokenizer:
            kw.pop("use_safetensors", None)
            return loader.from_pretrained(snapshot_dir, **kw)

        candidate = loader.from_pretrained(snapshot_dir, **kw)
        return activate_loaded_model(
            snapshot_dir,
            candidate,
            activation_key=activation_key,
            previous=previous,
        )
    except ArtifactRejected as exc:
        if previous is not None:
            warnings.warn(
                f"replacement for {model_id!r} was rejected; keeping the "
                f"previous model active: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return previous
        raise


# --------------------------------------------------------------------------- #
# 2. Pickle refusal (opcode-level, runs BEFORE any unpickling)                 #
# --------------------------------------------------------------------------- #
# The only globals a torch state_dict legitimately needs. Anything else - and
# anything we cannot statically resolve - is refused.
PICKLE_ALLOWED_GLOBALS = frozenset({
    "collections OrderedDict",
    "torch _utils _rebuild_tensor",
    "torch _utils _rebuild_tensor_v2",
    "torch _utils _rebuild_parameter",
    "torch FloatStorage", "torch DoubleStorage", "torch HalfStorage",
    "torch BFloat16Storage", "torch LongStorage", "torch IntStorage",
    "torch ShortStorage", "torch CharStorage", "torch ByteStorage",
    "torch BoolStorage",
    "numpy dtype", "numpy ndarray", "numpy core multiarray _reconstruct",
})

# Opcodes that build an object from a global we cannot see on the opcode stream.
_OPAQUE_CONSTRUCTORS = ("INST", "OBJ", "EXT1", "EXT2", "EXT4")


def _normalize_global(module: str, name: str) -> str:
    return f"{module.replace('.', ' ')} {name}"


def scan_pickle(data: bytes, *, where: str = "<bytes>") -> None:
    """Statically reject a pickle that references a non-allowlisted global.

    Walks the opcode stream with `pickletools.genops` - nothing is constructed
    and no module is imported, so this is safe to run on a hostile file. Raises
    `ArtifactRejected` on the first offending opcode.

    Both global opcodes are covered on purpose. Protocols 2 and 3 emit `GLOBAL`
    and protocols 4 and 5 emit `STACK_GLOBAL`; `pickle.dumps` defaults to 5, but
    an attacker picks the protocol, so covering only the default branch would
    leave the hole wide open (the negative control caught exactly that).

    Fail-closed in three places: an unresolvable STACK_GLOBAL, an opaque
    constructor opcode, and a stream that will not even parse are all refusals.
    """
    strings: list = []
    try:
        ops = list(pickletools.genops(data))
    except Exception as exc:  # truncated / malformed / adversarial framing
        raise ArtifactRejected(f"{where}: unparseable pickle stream ({exc})") from None

    for op, arg, _pos in ops:
        code = op.name
        if code in _OPAQUE_CONSTRUCTORS:
            raise ArtifactRejected(
                f"{where}: opcode {code} constructs an object from an "
                "unresolvable global; refused"
            )
        if code == "GLOBAL":
            mod, _, nm = str(arg).partition(" ")
            g = _normalize_global(mod, nm)
            if g not in PICKLE_ALLOWED_GLOBALS:
                raise ArtifactRejected(f"{where}: disallowed pickle global {arg!r}")
        elif code == "STACK_GLOBAL":
            if len(strings) < 2:
                raise ArtifactRejected(
                    f"{where}: STACK_GLOBAL with an unresolvable module/name pair"
                )
            mod, nm = strings[-2], strings[-1]
            g = _normalize_global(str(mod), str(nm))
            if g not in PICKLE_ALLOWED_GLOBALS:
                raise ArtifactRejected(
                    f"{where}: disallowed pickle global '{mod} {nm}'"
                )
            strings = strings[:-2]
        elif isinstance(arg, str):
            strings.append(arg)
            if len(strings) > 64:  # only the top of the stack can matter
                del strings[:-64]


def scan_checkpoint(path: str) -> None:
    """Run `scan_pickle` over every pickle inside a torch checkpoint file.

    Handles both the modern zip container (`archive/data.pkl`) and the legacy
    flat pickle stream.
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".pkl")]
            if not names:
                raise ArtifactRejected(f"{path}: zip checkpoint has no .pkl member")
            for n in names:
                scan_pickle(zf.read(n), where=f"{path}!{n}")
        return
    with open(path, "rb") as fh:
        scan_pickle(fh.read(), where=path)


def safe_load_state_dict(path: str) -> Mapping:
    """Load a state dict from `path`, preferring safetensors and never trusting
    a pickle.

    `.safetensors` is read directly (a pure tensor container, no code). A
    `.bin`/`.pt`/`.pth` is scanned with `scan_checkpoint` first and only then
    handed to `torch.load(..., weights_only=True)` - belt and braces, because
    `weights_only` is a property of the installed torch, and the scan is a
    property of this repo.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)
    if ext not in (".bin", ".pt", ".pth", ".ckpt"):
        raise ArtifactRejected(f"{path}: unsupported checkpoint extension {ext!r}")
    scan_checkpoint(path)
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


# --------------------------------------------------------------------------- #
# 3. Snapshot verification (digests + provenance), BEFORE activation           #
# --------------------------------------------------------------------------- #
def sha256_file(path: str, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(snapshot_dir: str, model_id: str, revision: str) -> dict:
    """Record a snapshot's provenance. Run this ONCE on a snapshot you have
    reviewed; commit the result and verify against it from then on."""
    files = {}
    for root, _dirs, names in os.walk(snapshot_dir):
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, snapshot_dir)
            files[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    return {"model_id": model_id, "revision": revision, "files": files}


def verify_snapshot(snapshot_dir: str, manifest: Mapping) -> None:
    """Reject a snapshot that does not match `manifest` exactly.

    Checks, in order: no executable Python in the tree, no missing files, no
    unexpected extra files, byte size, sha256. Every mismatch is a refusal -
    there is no "close enough" for a file we are about to load into a process.
    """
    expected = manifest["files"]
    present = {}
    for root, _dirs, names in os.walk(snapshot_dir):
        for n in names:
            p = os.path.join(root, n)
            present[os.path.relpath(p, snapshot_dir)] = p

    code = sorted(r for r in present if r.endswith(".py"))
    if code:
        raise ArtifactRejected(
            f"{snapshot_dir}: snapshot ships executable Python {code}; refused"
        )

    missing = sorted(set(expected) - set(present))
    if missing:
        raise ArtifactRejected(f"{snapshot_dir}: missing files {missing}")
    extra = sorted(set(present) - set(expected))
    if extra:
        raise ArtifactRejected(f"{snapshot_dir}: unexpected files {extra}")

    for rel in sorted(expected):
        want, path = expected[rel], present[rel]
        size = os.path.getsize(path)
        if size != want["bytes"]:
            raise ArtifactRejected(
                f"{snapshot_dir}: {rel} is {size} bytes, manifest says {want['bytes']}"
            )
        got = sha256_file(path)
        if got != want["sha256"]:
            raise ArtifactRejected(
                f"{snapshot_dir}: {rel} sha256 {got[:12]}... != manifest "
                f"{want['sha256'][:12]}..."
            )


# --------------------------------------------------------------------------- #
# 4. Tensor validation, BEFORE the weights are attached to a model             #
# --------------------------------------------------------------------------- #
def _dtype_name(t) -> str:
    return str(getattr(t, "dtype", "?")).replace("torch.", "")


def _all_finite(t) -> bool:
    isfinite = getattr(t, "isfinite", None)
    if isfinite is not None:  # torch / numpy tensor
        return bool(isfinite().all())
    values = t.tolist() if hasattr(t, "tolist") else t
    stack = [values]
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple)):
            stack.extend(v)
        elif v != v or v in (float("inf"), float("-inf")):
            return False
    return True


def validate_tensors(state: Mapping, spec: Mapping) -> None:
    """Reject a state dict that does not match `spec` before it is activated.

    `spec` is ``{"order": [name, ...], "tensors": {name: {"shape": [...],
    "dtype": "float32"}}}``. Key ORDER is checked as well as membership: the
    order is the feature order the downstream code indexes by, and a silently
    permuted state dict loads cleanly and then produces garbage.
    """
    want = spec["tensors"]
    got_keys = list(state.keys())

    missing = sorted(set(want) - set(got_keys))
    if missing:
        raise ArtifactRejected(f"state dict is missing tensors {missing}")
    extra = sorted(set(got_keys) - set(want))
    if extra:
        raise ArtifactRejected(f"state dict has unexpected tensors {extra}")

    order = spec.get("order")
    if order is not None and got_keys != list(order):
        raise ArtifactRejected(
            f"state dict key order differs from the spec: got {got_keys}, "
            f"expected {list(order)}"
        )

    for name in got_keys:
        t, w = state[name], want[name]
        shape = tuple(getattr(t, "shape", ()))
        if shape != tuple(w["shape"]):
            raise ArtifactRejected(
                f"{name}: shape {shape} != expected {tuple(w['shape'])}"
            )
        dtype = _dtype_name(t)
        if dtype != w["dtype"]:
            raise ArtifactRejected(f"{name}: dtype {dtype!r} != expected {w['dtype']!r}")
        if not _all_finite(t):
            raise ArtifactRejected(f"{name}: contains NaN or Inf")


# --------------------------------------------------------------------------- #
# 5. Known-good rollback                                                       #
# --------------------------------------------------------------------------- #
class KnownGood:
    """Last-verified manifest per model id, persisted as JSON.

    `activate` is the only way in: a candidate that fails verification raises
    and the store is left untouched, so `current(model_id)` still names the
    revision that was known to work.
    """

    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as fh:
                self._data = json.load(fh)

    def current(self, model_id: str) -> Optional[dict]:
        return self._data.get(model_id)

    def activate(
        self,
        snapshot_dir: str,
        manifest: Optional[Mapping] = None,
        *,
        state: Optional[Mapping] = None,
    ) -> dict:
        """Validate a candidate completely, then atomically promote its record.

        A `VerifiedSnapshot` is the production path. Its bytes already passed
        the Hugging Face manifest check, and `state` must now pass the persisted
        tensor schema plus the finite-value check. Plain paths retain the small
        generic-manifest path used by callers outside Hugging Face.
        """
        if isinstance(snapshot_dir, VerifiedSnapshot):
            if state is None:
                raise ArtifactRejected("verified model activation requires tensor state")
            model_id = snapshot_dir.model_id
            candidate_spec = _tensor_spec(state)
            current = self.current(model_id)
            expected_spec = current.get("tensor_spec") if current else None
            validate_tensors(state, expected_spec or candidate_spec)
            record = dict(snapshot_dir.manifest)
            record["model_id"] = model_id
            record["tensor_spec"] = candidate_spec
            return self.promote_verified(record)

        if state is not None:
            raise ArtifactRejected("tensor activation requires a VerifiedSnapshot")
        if manifest is None:
            raise ArtifactRejected("plain snapshot activation requires a manifest")
        verify_snapshot(snapshot_dir, manifest)
        return self.promote_verified(manifest)

    def promote_verified(self, record: Mapping) -> dict:
        """Persist a record after the caller completed artifact and tensor checks.

        This is intentionally separate from `activate`, whose generic manifest
        verifier does not understand Hugging Face git blob ids. Product code may
        call this only through `KnownGood.activate`.
        """
        model_id = record["model_id"]
        self._data[model_id] = dict(record)
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)  # atomic: never leave a half-written store
        return dict(record)


# --------------------------------------------------------------------------- #
# 6. Convenience wrappers for the Modal @enter loaders                         #
# --------------------------------------------------------------------------- #
def llmlingua_model_config(model_id: str, **extra) -> dict:
    """`model_config` for llmlingua's `PromptCompressor`, with remote code OFF.

    llmlingua 0.2.2 - the version `llmlingua2_modal.py` pins - defaults this to
    ON. From its own source (`llmlingua/prompt_compressor.py`, load_model):

        trust_remote_code = model_config.get("trust_remote_code", True)
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = trust_remote_code

    and the dict is then forwarded verbatim into `AutoConfig.from_pretrained`,
    `AutoTokenizer.from_pretrained` and `MODEL_CLASS.from_pretrained`. So unless
    the caller says otherwise the library imports and runs whatever Python the
    model directory ships - and here that directory is a mutable cache volume.
    Passing this dict turns it off and pins the revision.
    """
    cfg = {
        "trust_remote_code": False,
        "revision": pinned_revision(model_id),
        "local_files_only": True,
    }
    cfg.update(extra)
    if cfg.get("trust_remote_code"):
        raise ArtifactRejected(
            f"trust_remote_code=True on {model_id!r} executes repo-supplied "
            "Python at load time; refused"
        )
    if cfg.get("local_files_only") is not True:
        raise ArtifactRejected(
            f"local_files_only=False on {model_id!r} could bypass the verified "
            "snapshot; refused"
        )
    if cfg.get("revision") != pinned_revision(model_id):
        raise ArtifactRejected(
            f"{model_id!r} must use pinned revision {pinned_revision(model_id)}; "
            f"caller requested {cfg.get('revision')!r}"
        )
    return cfg


def pinned_snapshot_download(model_id: str, *, include_weights: bool = True, **kwargs):
    """`huggingface_hub.snapshot_download` at the pinned revision.

    Without a revision the download resolves `main`, so a cache that was
    populated last month and one populated today can hold different weights
    under the same name.
    """
    pinned = pinned_revision(model_id)
    requested = kwargs.get("revision")
    if requested is not None and requested != pinned:
        raise ArtifactRejected(
            f"{model_id!r} must use pinned revision {pinned}; "
            f"caller requested {requested!r}"
        )

    manifest = artifact_manifest(model_id)
    requested_patterns = kwargs.get("allow_patterns")
    expected_patterns = sorted(
        path for path in manifest["files"]
        if include_weights or ".safetensors" not in path
    )
    if requested_patterns is not None and sorted(requested_patterns) != expected_patterns:
        raise ArtifactRejected(
            f"{model_id!r} allow_patterns must exactly match the reviewed manifest"
        )

    from huggingface_hub import snapshot_download

    kwargs["revision"] = pinned
    kwargs["allow_patterns"] = expected_patterns
    return snapshot_download(model_id, **kwargs)


def artifact_manifest(model_id: str) -> dict:
    """Return the checked-in content manifest for one reviewed model."""
    try:
        from model_artifacts import ARTIFACT_MANIFEST

        manifest = ARTIFACT_MANIFEST["models"][model_id]
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactRejected(
            f"no readable artifact manifest for {model_id!r}: {exc}"
        ) from None
    pinned = pinned_revision(model_id)
    if manifest.get("revision") != pinned:
        raise ArtifactRejected(
            f"artifact manifest for {model_id!r} names revision "
            f"{manifest.get('revision')!r}, expected {pinned}"
        )
    return manifest


def _git_blob_oid(path: str) -> str:
    size = os.path.getsize(path)
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hf_snapshot(
    snapshot_dir: str,
    model_id: str,
    manifest: Optional[Mapping] = None,
    *,
    include_weights: bool = True,
) -> None:
    """Verify the exact files a Transformers loader may consume.

    The trust root is the checked-in manifest gathered from the pinned Hub
    commit. LFS entries carry their content sha256; regular git entries carry
    their git blob object id. Both are recomputed from local cache bytes before
    the cache path is passed to Transformers.
    """
    expected_revision = pinned_revision(model_id)
    manifest = dict(manifest or artifact_manifest(model_id))
    if manifest.get("revision") != expected_revision:
        raise ArtifactRejected(
            f"manifest revision {manifest.get('revision')!r} does not match "
            f"pinned revision {expected_revision}"
        )

    resolved = os.path.realpath(snapshot_dir)
    if os.path.basename(resolved) != expected_revision:
        raise ArtifactRejected(
            f"{snapshot_dir}: resolved snapshot is not pinned revision "
            f"{expected_revision}"
        )

    reviewed = manifest.get("files", {})
    expected = {
        path: digest for path, digest in reviewed.items()
        if include_weights or ".safetensors" not in path
    }
    for rel, wanted in expected.items():
        path = os.path.join(snapshot_dir, rel)
        if not os.path.isfile(path):
            raise ArtifactRejected(f"{snapshot_dir}: missing reviewed file {rel}")
        size = os.path.getsize(path)
        if size != wanted["bytes"]:
            raise ArtifactRejected(
                f"{snapshot_dir}: {rel} is {size} bytes, manifest says "
                f"{wanted['bytes']}"
            )
        if "sha256" in wanted:
            got = sha256_file(path)
            if got != wanted["sha256"]:
                raise ArtifactRejected(
                    f"{snapshot_dir}: {rel} sha256 {got[:12]}... != manifest "
                    f"{wanted['sha256'][:12]}..."
                )
        elif "git_oid" in wanted:
            got = _git_blob_oid(path)
            if got != wanted["git_oid"]:
                raise ArtifactRejected(
                    f"{snapshot_dir}: {rel} git blob {got[:12]}... != manifest "
                    f"{wanted['git_oid'][:12]}..."
                )
        else:
            raise ArtifactRejected(f"{rel}: manifest entry has no digest")

    loadable_suffixes = (
        ".json", ".txt", ".model", ".tiktoken", ".safetensors",
        ".bin", ".pt", ".pth", ".ckpt", ".py",
    )
    for root, _dirs, names in os.walk(snapshot_dir):
        for name in names:
            rel = os.path.relpath(os.path.join(root, name), snapshot_dir)
            if rel.endswith(loadable_suffixes) and rel not in reviewed:
                raise ArtifactRejected(
                    f"{snapshot_dir}: unreviewed loadable file {rel}; refused"
                )
    if include_weights:
        assert_no_pickled_weights(snapshot_dir)


def verified_snapshot_download(
    model_id: str, *, include_weights: bool = True, **kwargs
) -> VerifiedSnapshot:
    """Download only reviewed files, verify their bytes, then return the path."""
    snapshot_dir = pinned_snapshot_download(
        model_id, include_weights=include_weights, **kwargs
    )
    verify_hf_snapshot(snapshot_dir, model_id, include_weights=include_weights)
    return VerifiedSnapshot(snapshot_dir, model_id, artifact_manifest(model_id))


def _tensor_spec(state: Mapping) -> dict:
    if not state:
        raise ArtifactRejected("loaded model exposes an empty state dict")
    order = list(state.keys())
    tensors = {}
    for name in order:
        tensor = state[name]
        shape = tuple(getattr(tensor, "shape", ()))
        if not shape:
            raise ArtifactRejected(f"{name}: tensor has no declared shape")
        tensors[name] = {
            "shape": [int(value) for value in shape],
            "dtype": _dtype_name(tensor),
        }
    return {"order": order, "tensors": tensors}


def _known_good_path() -> str:
    configured = os.environ.get("WINNOW_KNOWN_GOOD_PATH")
    if configured:
        return configured
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(hf_home, "winnow-known-good.json")


def activate_loaded_model(
    snapshot: VerifiedSnapshot,
    candidate,
    *,
    tensor_owner=None,
    activation_key=None,
    previous=None,
):
    """Validate loaded tensors, persist known-good state, then publish candidate.

    `snapshot` can only come from `verified_snapshot_download`. `tensor_owner`
    handles wrappers such as LLMLingua whose actual torch module is `.model`.
    A rejected replacement returns the prior in-process object when one exists;
    otherwise it fails closed. The persisted known-good record is never changed
    by a rejected candidate, so a cold process retains an operator-visible
    rollback target even though Python model objects themselves are not portable.
    """
    if not isinstance(snapshot, VerifiedSnapshot):
        raise ArtifactRejected("activation requires a verified snapshot")
    model_id = snapshot.model_id
    key = activation_key or (model_id, type(candidate).__module__, type(candidate).__name__)
    previous = previous if previous is not None else _ACTIVE_MODELS.get(key)
    source = tensor_owner if tensor_owner is not None else candidate

    try:
        state_dict = getattr(source, "state_dict", None)
        if not callable(state_dict):
            raise ArtifactRejected("loaded model exposes no state_dict for validation")
        state = state_dict()
        known_good = KnownGood(_known_good_path())
        known_good.activate(snapshot, state=state)
    except ArtifactRejected as exc:
        if previous is not None:
            warnings.warn(
                f"replacement for {model_id!r} was rejected; keeping the "
                f"previous model active: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return previous
        raise

    _ACTIVE_MODELS[key] = candidate
    return candidate


def assert_no_pickled_weights(snapshot_dir: str) -> Sequence[str]:
    """Refuse a snapshot that still contains a pickled checkpoint next to the
    safetensors one (e.g. BAAI/bge-small-en-v1.5 ships both). Returns the
    safetensors files it found."""
    pickled, safe = [], []
    for root, _dirs, names in os.walk(snapshot_dir):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), snapshot_dir)
            if n.endswith((".bin", ".pt", ".pth", ".ckpt")):
                pickled.append(rel)
            elif n.endswith(".safetensors"):
                safe.append(rel)
    if pickled:
        raise ArtifactRejected(
            f"{snapshot_dir}: pickled checkpoint(s) present {sorted(pickled)}; "
            "delete them or download with allow_patterns that exclude them"
        )
    if not safe:
        raise ArtifactRejected(f"{snapshot_dir}: no safetensors weights found")
    return sorted(safe)
