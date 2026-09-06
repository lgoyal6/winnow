"""Local tests for model_guard (no GPU, no torch, no network).

Run: python test_model_guard.py

Two halves:
  * the guard itself - pickle refusal, snapshot digests, tensor validation,
    known-good rollback;
  * the WIRING - a fake `transformers` is injected into sys.modules and the real
    loader classes are constructed, so the tests assert what the production call
    sites actually pass to `from_pretrained`. The sites that cannot be
    constructed on a CPU box (the Modal `@enter` loaders, the `turboquant_kv`
    benchmark scripts) are covered by the repo-wide AST sweep instead.
"""

import collections
import contextlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import types
import zipfile

from model_guard import (
    ArtifactRejected,
    KnownGood,
    assert_no_pickled_weights,
    build_manifest,
    guarded_kwargs,
    llmlingua_model_config,
    pinned_revision,
    scan_checkpoint,
    scan_pickle,
    validate_tensors,
    verify_snapshot,
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
class FakeTensor:
    """Minimal tensor stand-in: shape + dtype + tolist(), no torch needed."""

    def __init__(self, shape, dtype="float32", values=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        n = 1
        for d in self.shape:
            n *= d
        self._values = list(values) if values is not None else [0.5] * n

    def tolist(self):
        return self._values


class _Payload:
    """Pickling this yields a stream that CALLS os.makedirs on load."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (os.makedirs, (self.path,))


@contextlib.contextmanager
def _tmpdir():
    d = tempfile.mkdtemp(prefix="winnow-guard-")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__} from {getattr(fn, '__name__', fn)}")


# --------------------------------------------------------------------------- #
# 1. the pickle RCE primitive, and its refusal                                 #
# --------------------------------------------------------------------------- #
def test_malicious_pickle_really_executes_but_is_refused():
    """The primitive is real: prove it fires, then prove the scanner stops it.

    Half one runs the payload through plain `pickle.loads` and checks the
    side effect actually happened - without that half, "the scanner refused it"
    proves nothing. Half two shows `scan_pickle` refusing the identical bytes
    with no unpickling at all.
    """
    with _tmpdir() as d:
        canary = os.path.join(d, "pwned")
        blob = pickle.dumps(_Payload(canary))

        assert not os.path.exists(canary)
        pickle.loads(blob)  # <- the RCE primitive, unguarded
        assert os.path.isdir(canary), "unguarded pickle.loads did NOT execute"

        canary2 = os.path.join(d, "pwned2")
        blob2 = pickle.dumps(_Payload(canary2))
        msg = _raises(ArtifactRejected, scan_pickle, blob2, where="payload")
        assert "makedirs" in msg, msg
        assert not os.path.exists(canary2), "scan_pickle must not execute anything"


def test_malicious_pickle_is_refused_at_every_protocol():
    """Protocol 2/3 emit GLOBAL, protocol 4/5 emit STACK_GLOBAL - the scanner
    has to cover both branches, and an attacker picks the protocol.

    (This test exists because the negative control found it missing: `pickle.dumps`
    defaults to protocol 5, so disabling the GLOBAL branch alone left the suite
    green.)
    """
    import pickletools

    seen = set()
    for proto in (2, 3, 4, 5):
        blob = pickle.dumps(_Payload("/tmp/winnow-guard-never-created"), protocol=proto)
        seen.update(op.name for op, _a, _p in pickletools.genops(blob)
                    if op.name in ("GLOBAL", "STACK_GLOBAL"))
        msg = _raises(ArtifactRejected, scan_pickle, blob, where=f"proto{proto}")
        assert "makedirs" in msg, (proto, msg)
    assert seen == {"GLOBAL", "STACK_GLOBAL"}, f"both branches must be exercised: {seen}"
    assert not os.path.exists("/tmp/winnow-guard-never-created")


def test_benign_state_dict_pickle_passes():
    blob = pickle.dumps(collections.OrderedDict([("a.weight", [1.0, 2.0])]))
    scan_pickle(blob)  # must not raise


def test_malicious_pickle_inside_a_torch_zip_checkpoint_is_refused():
    """A real `.bin` is a zip with `archive/data.pkl` inside; scan every member."""
    with _tmpdir() as d:
        ckpt = os.path.join(d, "pytorch_model.bin")
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps(_Payload(os.path.join(d, "x"))))
            zf.writestr("archive/data/0", b"\x00" * 8)
        msg = _raises(ArtifactRejected, scan_checkpoint, ckpt)
        assert "data.pkl" in msg and "makedirs" in msg, msg


def test_truncated_pickle_is_refused_not_ignored():
    blob = pickle.dumps(collections.OrderedDict([("a", 1)]))
    _raises(ArtifactRejected, scan_pickle, blob[: len(blob) // 2])


def test_zip_without_a_pickle_member_is_refused():
    with _tmpdir() as d:
        ckpt = os.path.join(d, "weights.bin")
        with zipfile.ZipFile(ckpt, "w") as zf:
            zf.writestr("archive/data/0", b"\x00")
        _raises(ArtifactRejected, scan_checkpoint, ckpt)


# --------------------------------------------------------------------------- #
# 2. provenance and load flags                                                 #
# --------------------------------------------------------------------------- #
def test_pinned_revision_is_a_sha_and_unknown_ids_are_refused():
    rev = pinned_revision("Qwen/Qwen2.5-7B-Instruct")
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), rev
    _raises(ArtifactRejected, pinned_revision, "attacker/backdoored-model")


def test_every_model_id_used_in_the_repo_has_a_pin():
    """A model id that appears in the source but not in REVISIONS is a load site
    that would be refused at runtime instead of at review time."""
    import re

    ids = set()
    pat = re.compile(r'["\']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)["\']')
    for path in _repo_py_files():
        with open(path) as fh:
            for m in pat.finditer(fh.read()):
                cand = m.group(1)
                if cand.split("/")[0] in ("microsoft", "BAAI", "Qwen", "mistralai"):
                    ids.add(cand)
    missing = sorted(i for i in ids if i not in __import__("model_guard").REVISIONS)
    assert not missing, f"model ids used in the repo with no pinned revision: {missing}"


def test_guarded_kwargs_forces_the_flags_and_refuses_opt_out():
    kw = guarded_kwargs("BAAI/bge-small-en-v1.5")
    assert kw["revision"] == pinned_revision("BAAI/bge-small-en-v1.5")
    assert kw["use_safetensors"] is True
    assert kw["trust_remote_code"] is False
    _raises(ArtifactRejected, guarded_kwargs, "BAAI/bge-small-en-v1.5",
            trust_remote_code=True)
    _raises(ArtifactRejected, guarded_kwargs, "BAAI/bge-small-en-v1.5",
            use_safetensors=False)


def test_llmlingua_model_config_turns_remote_code_off_and_pins():
    """llmlingua 0.2.2 defaults trust_remote_code to True; this is the opt-out."""
    cfg = llmlingua_model_config("microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    assert cfg["trust_remote_code"] is False
    assert cfg["revision"] == pinned_revision(
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    _raises(ArtifactRejected, llmlingua_model_config,
            "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            trust_remote_code=True)


# --------------------------------------------------------------------------- #
# 3. snapshot verification BEFORE activation                                   #
# --------------------------------------------------------------------------- #
def _good_snapshot(root):
    _write(os.path.join(root, "config.json"), b'{"hidden_size": 8}')
    _write(os.path.join(root, "model.safetensors"), b"tensor-bytes")
    return build_manifest(root, "BAAI/bge-small-en-v1.5",
                          pinned_revision("BAAI/bge-small-en-v1.5"))


def test_verify_snapshot_accepts_the_snapshot_it_was_built_from():
    with _tmpdir() as d:
        verify_snapshot(d, _good_snapshot(d))


def test_verify_snapshot_refuses_a_tampered_file():
    with _tmpdir() as d:
        m = _good_snapshot(d)
        _write(os.path.join(d, "model.safetensors"), b"tensor-bytez")  # 1 byte flipped
        msg = _raises(ArtifactRejected, verify_snapshot, d, m)
        assert "sha256" in msg, msg


def test_verify_snapshot_refuses_missing_and_extra_files():
    with _tmpdir() as d:
        m = _good_snapshot(d)
        _write(os.path.join(d, "surprise.json"), b"{}")
        assert "unexpected files" in _raises(ArtifactRejected, verify_snapshot, d, m)
        os.remove(os.path.join(d, "surprise.json"))
        os.remove(os.path.join(d, "config.json"))
        assert "missing files" in _raises(ArtifactRejected, verify_snapshot, d, m)


def test_verify_snapshot_refuses_a_snapshot_shipping_python():
    with _tmpdir() as d:
        m = _good_snapshot(d)
        _write(os.path.join(d, "modeling_custom.py"), b"import os\n")
        msg = _raises(ArtifactRejected, verify_snapshot, d, m)
        assert "executable Python" in msg, msg


def test_assert_no_pickled_weights():
    with _tmpdir() as d:
        _write(os.path.join(d, "model.safetensors"), b"ok")
        assert assert_no_pickled_weights(d) == ["model.safetensors"]
        _write(os.path.join(d, "pytorch_model.bin"), b"pickled")
        msg = _raises(ArtifactRejected, assert_no_pickled_weights, d)
        assert "pytorch_model.bin" in msg, msg


# --------------------------------------------------------------------------- #
# 4. tensor validation BEFORE activation                                       #
# --------------------------------------------------------------------------- #
_SPEC = {
    "order": ["enc.weight", "enc.bias"],
    "tensors": {
        "enc.weight": {"shape": [2, 3], "dtype": "float32"},
        "enc.bias": {"shape": [2], "dtype": "float32"},
    },
}


def _good_state():
    return collections.OrderedDict([
        ("enc.weight", FakeTensor([2, 3])),
        ("enc.bias", FakeTensor([2])),
    ])


def test_validate_tensors_accepts_a_matching_state_dict():
    validate_tensors(_good_state(), _SPEC)


def test_validate_tensors_refuses_wrong_shape_dtype_and_missing_key():
    s = _good_state()
    s["enc.bias"] = FakeTensor([3])
    assert "shape" in _raises(ArtifactRejected, validate_tensors, s, _SPEC)

    s = _good_state()
    s["enc.bias"] = FakeTensor([2], dtype="int8")
    assert "dtype" in _raises(ArtifactRejected, validate_tensors, s, _SPEC)

    s = _good_state()
    del s["enc.bias"]
    assert "missing tensors" in _raises(ArtifactRejected, validate_tensors, s, _SPEC)

    s = _good_state()
    s["enc.extra"] = FakeTensor([1])
    assert "unexpected tensors" in _raises(ArtifactRejected, validate_tensors, s, _SPEC)


def test_validate_tensors_refuses_non_finite_values():
    s = _good_state()
    s["enc.bias"] = FakeTensor([2], values=[0.1, float("nan")])
    assert "NaN or Inf" in _raises(ArtifactRejected, validate_tensors, s, _SPEC)
    s["enc.bias"] = FakeTensor([2], values=[0.1, float("inf")])
    assert "NaN or Inf" in _raises(ArtifactRejected, validate_tensors, s, _SPEC)


def test_validate_tensors_refuses_permuted_feature_order():
    """Same keys, same shapes, wrong order - loads fine, then produces garbage."""
    s = collections.OrderedDict([
        ("enc.bias", FakeTensor([2])),
        ("enc.weight", FakeTensor([2, 3])),
    ])
    assert "key order" in _raises(ArtifactRejected, validate_tensors, s, _SPEC)


# --------------------------------------------------------------------------- #
# 5. known-good rollback                                                       #
# --------------------------------------------------------------------------- #
def test_rejected_candidate_leaves_the_previous_known_good_in_place():
    with _tmpdir() as d:
        store_path = os.path.join(d, "known_good.json")
        good_dir = os.path.join(d, "v1")
        good = _good_snapshot(good_dir)

        store = KnownGood(store_path)
        store.activate(good_dir, good)
        assert store.current(good["model_id"])["revision"] == good["revision"]

        # a tampered candidate at the same path must not overwrite the record
        bad_dir = os.path.join(d, "v2")
        bad = _good_snapshot(bad_dir)
        bad["revision"] = "0" * 40
        _write(os.path.join(bad_dir, "model.safetensors"), b"tampered")
        _raises(ArtifactRejected, store.activate, bad_dir, bad)

        assert store.current(good["model_id"])["revision"] == good["revision"]
        reopened = KnownGood(store_path)  # and it survived on disk
        assert reopened.current(good["model_id"])["revision"] == good["revision"]
        assert json.load(open(store_path))[good["model_id"]]["revision"] == good["revision"]


# --------------------------------------------------------------------------- #
# 6. WIRING: what the real load sites pass to from_pretrained                  #
# --------------------------------------------------------------------------- #
def _install_fake_transformers():
    """Fake torch + transformers so the real loader classes can be constructed
    on a CPU-only box. Returns the list every from_pretrained call is recorded
    into."""
    calls = []

    torch = types.ModuleType("torch")
    torch.float16 = "torch.float16"
    torch.bfloat16 = "torch.bfloat16"
    nn = types.ModuleType("torch.nn")
    nn.functional = types.ModuleType("torch.nn.functional")
    torch.nn = nn
    sys.modules.update({"torch": torch, "torch.nn": nn,
                        "torch.nn.functional": nn.functional})

    class _Loaded:
        config = types.SimpleNamespace(num_hidden_layers=2)

        def eval(self):
            return self

        def half(self):
            return self

        def to(self, *a, **k):
            return self

    def _auto(name):
        return type(name, (), {
            "from_pretrained": classmethod(
                lambda cls, model_id, **kw: (calls.append((name, model_id, kw)), _Loaded())[1]
            )
        })

    tf = types.ModuleType("transformers")
    for n in ("AutoTokenizer", "AutoModel", "AutoModelForCausalLM",
              "AutoModelForSequenceClassification"):
        setattr(tf, n, _auto(n))
    sys.modules["transformers"] = tf
    return calls


def _assert_guarded(calls, expect_n):
    assert len(calls) == expect_n, f"expected {expect_n} loads, saw {len(calls)}"
    for name, model_id, kw in calls:
        assert kw.get("revision") == pinned_revision(model_id), \
            f"{name}({model_id}) not pinned: revision={kw.get('revision')!r}"
        assert kw.get("trust_remote_code") is False, \
            f"{name}({model_id}) did not disable remote code"
        if "Tokenizer" not in name:
            assert kw.get("use_safetensors") is True, \
                f"{name}({model_id}) did not force safetensors"


def test_two_stage_compressor_load_sites_are_guarded():
    calls = _install_fake_transformers()
    import two_stage_compressor as tsc

    tsc.SmallEmbedder(device="cpu", use_fp16=False)
    tsc.CrossEncoderReranker("BAAI/bge-reranker-v2-m3", device="cpu", use_fp16=False)
    _assert_guarded(calls, 4)


def test_attentionrag_hf_backend_load_sites_are_guarded():
    calls = _install_fake_transformers()
    from attentionrag.hf_backend import HFBackend

    HFBackend(device="cpu")
    _assert_guarded(calls, 2)


# --------------------------------------------------------------------------- #
# 7. STATIC SWEEP: no unguarded artifact load anywhere in the repo             #
# --------------------------------------------------------------------------- #
_REPO = os.path.dirname(os.path.abspath(__file__))
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".next", ".agent-work",
              ".venv", "venv", "dist", "build"}


def _repo_py_files():
    for root, dirs, names in os.walk(_REPO):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for n in sorted(names):
            # model_guard.py holds the ONLY sanctioned raw calls (they are the
            # wrappers); this test file only names them in assertions.
            if n.endswith(".py") and n not in (os.path.basename(__file__),
                                               "model_guard.py"):
                yield os.path.join(root, n)


def test_no_unguarded_from_pretrained_or_snapshot_download_in_repo():
    """A new load site added without the guard is a regression, so fail on it.

    Catches `X.from_pretrained(...)` (must be `guarded_from_pretrained`), bare
    `snapshot_download(...)` (must be `pinned_snapshot_download`), and a
    `PromptCompressor(...)` without `model_config` (llmlingua 0.2.2 defaults
    trust_remote_code to True).
    """
    import ast

    offenders = []
    for path in _repo_py_files():
        rel = os.path.relpath(path, _REPO)
        with open(path) as fh:
            src = fh.read()
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "from_pretrained":
                offenders.append(f"{rel}:{node.lineno} bare .from_pretrained()")
            elif isinstance(fn, ast.Name):
                if fn.id == "snapshot_download":
                    offenders.append(f"{rel}:{node.lineno} bare snapshot_download()")
                elif fn.id == "PromptCompressor" and not any(
                    kw.arg == "model_config" for kw in node.keywords
                ):
                    offenders.append(
                        f"{rel}:{node.lineno} PromptCompressor() without model_config"
                    )
    assert not offenders, "unguarded artifact loads:\n  " + "\n  ".join(offenders)


def test_every_pinned_snapshot_download_is_wrapped_in_assert_no_pickled_weights():
    """`pinned_snapshot_download` pins the revision but does NOT inspect what it
    downloaded, so on its own it still lands pickled weights on the cache volume.
    Every call must be the argument of `assert_no_pickled_weights(...)`.

    This is a real regression that has happened: in the TC960/winnow checkout,
    `experiments/eval_modal.py` downloads three models in a loop with a bare
    `pinned_snapshot_download(name)`, so the pickle check was skipped for all
    three while the other two Modal apps in the same repo did wrap it. The
    previous static sweep does not catch that shape, because the call itself is
    the guarded wrapper. This test does.
    """
    import ast

    offenders = []
    for path in _repo_py_files():
        rel = os.path.relpath(path, _REPO)
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=rel)

        wrapped = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "assert_no_pickled_weights"):
                for arg in node.args:
                    wrapped.add(id(arg))

        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "pinned_snapshot_download"
                    and id(node) not in wrapped):
                offenders.append(
                    f"{rel}:{node.lineno} pinned_snapshot_download() not wrapped in "
                    "assert_no_pickled_weights()")
    assert not offenders, ("snapshot downloads whose contents are never checked:\n  "
                           + "\n  ".join(offenders))


def test_no_bare_torch_load_or_pickle_load_in_repo():
    """`torch.load` / `pickle.load(s)` unpickle before anything is validated.

    None exist today; this test is the tripwire that keeps it that way, since a
    single `torch.load` on a cache-volume file would reopen the whole hole that
    `safe_load_state_dict` exists to close.
    """
    import ast

    offenders = []
    for path in _repo_py_files():
        rel = os.path.relpath(path, _REPO)
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = getattr(node.func.value, "id", None)
            if owner == "torch" and node.func.attr == "load":
                offenders.append(f"{rel}:{node.lineno} torch.load()")
            elif owner == "pickle" and node.func.attr in ("load", "loads"):
                offenders.append(f"{rel}:{node.lineno} pickle.{node.func.attr}()")
    assert not offenders, ("unpickling outside model_guard.safe_load_state_dict:\n  "
                           + "\n  ".join(offenders))


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} model_guard tests passed.")


if __name__ == "__main__":
    _run_all()
