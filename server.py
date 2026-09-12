"""
FastAPI server in front of the Modal compression + generation workers.

All GPU workers are tied to THIS server's lifecycle (identical mechanism):
  * on startup  -> each Modal app starts (`app.run()`) and its container
                   cold-starts and loads its model (we warm each once), so real
                   requests have no model-startup cost.
  * on shutdown -> the Modal apps stop, tearing down their GPU containers
                   immediately (GPUs released; no idle lingering).

Workers:
  * LLMLingua-2 (Compressor)              -> /compress, /compress_rag   (A100)
  * AttentionRAG (AttentionRAGService)    -> /compress (when `question` set; A100)
  * TurboQuant  (TurboQuantModel)         -> /generate (default route)  (A100-80GB)
  * LCLM+TurboQuant (LCLMTurboQuantModel) -> /generate (lclm=true)      (A100-80GB)

/compress picks behavior by the request's `question`:
  * question empty (default) -> LLMLingua-2 token compression only (back-compat).
  * question set             -> run LLMLingua-2 AND AttentionRAG in parallel and
                                MERGE the two keep-decisions token-by-token over the
                                original text (intersection or union).

/generate picks a worker by the request's `lclm` flag:
  * lclm=false (default) -> Qwen TurboQuant route (KV-cache bit quantization).
  * lclm=true            -> LCLM (encoder-decoder context compression) + TurboQuant
                            on the decoder. Pass the long context in `context`; it
                            is compressed into latent soft tokens, while `prompt`
                            (the question/instruction) stays verbatim.

Prereqs:
    pip install fastapi "uvicorn[standard]" pydantic modal torch transformers ...

Run (no --reload: the lifespan owns the Modal apps; reload would double-start them):
    uvicorn server:app --port 8000

Try it (Qwen TurboQuant route, default):
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Explain KV-cache quantization.", "bit_width": 4, "max_new_tokens": 120}'

Try it (LCLM + TurboQuant route):
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"lclm": true, "prompt": "What is the calibration passphrase?", "context": "your long document with a planted fact ...", "bit_width": 4, "max_new_tokens": 120}'

    # plain token-level compression of one blob of text:
    curl -X POST http://localhost:8000/compress \
        -H "Content-Type: application/json" \
        -d '{"text": "your long text here ...", "rate": 0.5}'

    # question-aware: LLMLingua + AttentionRAG merged over the original text:
    curl -X POST http://localhost:8000/compress \
        -H "Content-Type: application/json" \
        -d '{"text": "your long text ...", "question": "What is X?", "mode": "intersection", "return_labels": true}'
"""

import asyncio
import time
from contextlib import ExitStack, asynccontextmanager
from typing import Any, Callable, List, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from lifecycle import ComponentSpec, ComponentUnavailable, Lifecycle

# Token-by-token merge of LLMLingua + AttentionRAG keep-decisions (pure-python).
from token_merge import merge_compress, normalize_labels

# Black-box downstream LLM callers (Claude / ChatGPT) reused for the playground.
from downstream import (
    DEFAULT_MODEL as BLACKBOX_DEFAULT_MODEL,
    _call_claude,
    _call_openai,
    _canonical_provider,
    _key_for,
)

# The four workers, and which of them this service cannot serve without.
#
# Critical: LLMLingua-2 is the whole of /compress, and the default TurboQuant
# route is the whole of /generate. Without either there is no service to run.
#
# Optional: AttentionRAG makes /compress question-aware and LCLM adds a second
# generation route. Losing one of those costs one feature; taking the process
# down over it costs all of them.
COMPRESSOR = "compressor"
ATTENTIONRAG = "attentionrag"
TURBOQUANT = "turboquant"
LCLM = "lclm"

LIFECYCLE: Lifecycle | None = None
"""Set by the lifespan. Read through `require`, never directly: an endpoint that
reached for a handle itself would be the thing that used to serve requests from
a worker with no model in it."""


def _modal_component(
    *,
    name: str,
    capability: str,
    critical: bool,
    load: Callable[[], tuple[Any, Any]],
    warm: Callable[[Any], Any],
) -> ComponentSpec:
    """One Modal worker, behind three plain callables.

    Every Modal-shaped thing in this service is on this side of the line:
    `app.run()`, the class handle, and the `.remote.aio` warmup. The lifecycle
    itself has no idea Modal exists, which is what lets the tests drive startup,
    rollback, degradation and shutdown with fakes and never open a connection.

    `app.run()` is a synchronous context manager, so it is entered on a worker
    thread rather than blocking the event loop while a container comes up.
    """
    stack = ExitStack()

    async def start() -> Any:
        app_handle, worker = await asyncio.to_thread(load_and_enter)
        return worker

    def load_and_enter() -> tuple[Any, Any]:
        modal_app, worker_factory = load()
        stack.enter_context(modal_app.run())
        return modal_app, worker_factory()

    async def warmup(worker: Any) -> Any:
        return await warm(worker)

    async def cleanup(_worker: Any) -> None:
        await asyncio.to_thread(stack.close)

    return ComponentSpec(
        name=name,
        capability=capability,
        critical=critical,
        start=start,
        warmup=warmup,
        cleanup=cleanup,
    )


def modal_components() -> list[ComponentSpec]:
    """The real worker set.

    The worker modules are imported here rather than at module scope so that
    importing `server` costs nothing and reaches nothing. That is not tidiness:
    it is what makes it possible to test every endpoint without `modal`
    installed, and it removes any path by which a test run could contact it.
    """
    import attentionrag.modal_app as attentionrag_modal
    import lclm_worker_modal
    import llmlingua2_modal
    import turboquant_modal

    return [
        _modal_component(
            name=COMPRESSOR,
            capability="LLMLingua-2 token compression",
            critical=True,
            load=lambda: (llmlingua2_modal.app, llmlingua2_modal.Compressor),
            warm=lambda w: w.compress.remote.aio("warmup", rate=0.5),
        ),
        _modal_component(
            name=ATTENTIONRAG,
            capability="AttentionRAG question-aware selection",
            critical=False,
            load=lambda: (attentionrag_modal.app, attentionrag_modal.AttentionRAGService),
            warm=lambda w: w.compress_spans.remote.aio("warmup", "warmup"),
        ),
        _modal_component(
            name=TURBOQUANT,
            capability="TurboQuant generation",
            critical=True,
            load=lambda: (turboquant_modal.app, turboquant_modal.TurboQuantModel),
            warm=lambda w: w.generate.remote.aio("warmup", max_new_tokens=1),
        ),
        _modal_component(
            name=LCLM,
            capability="LCLM long-context generation",
            critical=False,
            load=lambda: (lclm_worker_modal.app, lclm_worker_modal.LCLMTurboQuantModel),
            warm=lambda w: w.generate.remote.aio("warmup", max_new_tokens=1),
        ),
    ]


component_factory: Callable[[], list[ComponentSpec]] = modal_components
"""What builds the worker set. Swapped for fakes in tests, which is the only
reason it is a name rather than a call. Not a switch over behaviour: whatever it
returns goes through exactly the same startup, rollback and readiness path."""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start the workers when the server boots; stop them when it exits.

    `app.run()` starts an EPHEMERAL Modal app bound to this process: its
    containers live only while this server lives, and each worker is warmed once
    so the cold start happens here rather than on somebody's first request.

    What changed is what happens when one of them does not come up. A critical
    worker failing still aborts the boot, and everything already started is
    closed on the way out. An optional worker failing is recorded, cleaned up,
    and the service serves everything that does not need it -- see /ready for
    which of those is currently true.
    """
    global LIFECYCLE
    LIFECYCLE = Lifecycle(component_factory())
    try:
        await LIFECYCLE.start()
    except Exception:
        LIFECYCLE = None
        raise
    missing = [c for c in LIFECYCLE.readiness()["components"] if not c["ready"]]
    if missing:
        print(
            "[startup] serving degraded, unavailable: "
            + ", ".join(f"{c['capability']} ({c['state']})" for c in missing),
            flush=True,
        )
    else:
        print("[startup] all workers warm; ready to serve.", flush=True)

    try:
        yield  # ----------------- server handles requests -----------------
    finally:
        await LIFECYCLE.stop()
        LIFECYCLE = None
        print("[shutdown] workers stopped; GPU containers released.", flush=True)


app = FastAPI(title="Compression + Generation API", lifespan=lifespan)


def require(name: str) -> Any:
    """The worker behind a component, or a 503 that names what is missing.

    A 503 rather than a quiet substitution. Answering a request for LCLM with
    the default generation route would return a plausible answer from a model
    the caller did not ask for, and nothing in the response would say so.
    """
    if LIFECYCLE is None:
        raise HTTPException(status_code=503, detail="the service is still starting")
    try:
        return LIFECYCLE.handle(name)
    except ComponentUnavailable as unavailable:
        raise HTTPException(status_code=503, detail=str(unavailable)) from unavailable


# --------------------------------------------------------------------------- #
# LLMLingua-2 (+ optional AttentionRAG merge) compression
# --------------------------------------------------------------------------- #
class CompressRequest(BaseModel):
    text: str = Field(..., description="Text to compress")
    rate: float = Field(0.5, gt=0, le=1, description="LLMLingua fraction of tokens to keep")
    return_labels: bool = Field(False, description="Include per-word keep/discard labels")
    # --- AttentionRAG + merge controls -----------------------------------
    # When `question` is set, AttentionRAG runs in parallel with LLMLingua and
    # the two keep-decisions are merged token-by-token over the original text.
    # When `question` is empty, only LLMLingua runs (back-compat behavior).
    question: Optional[str] = Field(
        None, description="Query for AttentionRAG; enables the parallel merge"
    )
    mode: str = Field(
        "intersection", description="'intersection' (both keep) or 'union' (either keeps)"
    )
    chunk_size: int = Field(300, gt=0, description="AttentionRAG chunk size (tokens)")
    top_k: int = Field(12, gt=0, description="AttentionRAG top-k tokens per chunk")
    use_openai_hint: bool = Field(
        False, description="Author AttentionRAG hint prefix with GPT-4o-mini"
    )


class CompressResponse(BaseModel):
    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    rate: float | str  # LLMLingua may return a percentage string e.g. '47.6%'
    ratio: str
    # Per-word keep/drop labels: list of (word, 1|0). When a merge ran these are
    # the MERGED labels; otherwise LLMLingua's. Powers the strike-through diff UI.
    word_labels: Optional[list] = None
    # --- merge diagnostics (populated only when a merge ran) --------------
    mode: Optional[str] = None
    merged: bool = False
    used_llmlingua_fallback: Optional[bool] = None
    words_total: Optional[int] = None
    words_kept: Optional[int] = None
    merged_ratio: Optional[str] = None
    attentionrag_hint_prefix: Optional[str] = None
    attentionrag_kept_chunks: Optional[str] = None


class RagRequest(BaseModel):
    instruction: str = Field("", description="System/task instruction (kept verbatim)")
    question: str = Field(..., description="User query (drives ranking, kept verbatim)")
    documents: List[str] = Field(..., description="Retrieved chunks, one per element")
    rate: float = Field(0.5, gt=0, le=1, description="Fine-stage fraction of tokens to keep")
    target_token: int = Field(-1, description="Hard token budget for the context (-1 = use rate)")
    top_k: Optional[int] = Field(None, description="Max documents to keep in the coarse stage")
    score_threshold: Optional[float] = Field(
        None, description="Min reranker score [0,1] to keep a document"
    )


def _parse_rate(v) -> float:
    """LLMLingua returns `rate` as a display string like '45.0%'; coerce to a
    float fraction in [0,1] for the response model."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    pct = s.endswith("%")
    try:
        f = float(s.rstrip("%"))
    except ValueError:
        return 0.0
    return f / 100.0 if pct else f


# --------------------------------------------------------------------------- #
# TurboQuant generation
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to generate from")
    bit_width: int = Field(4, ge=2, le=8, description="TurboQuant bits per KV value")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="Max tokens to generate")
    outlier_channels: int = Field(
        0, ge=0, description="Per-head channels kept at higher precision (0 = off)"
    )
    outlier_bits: int = Field(
        0, ge=0, description="Bits for outlier channels (must exceed bit_width to take effect)"
    )
    lclm: bool = Field(
        False,
        description="Route through the LCLM (context-compression) + TurboQuant worker "
                    "instead of the default Qwen TurboQuant worker.",
    )
    context: str = Field(
        "",
        description="LCLM-only: long context to compress into latent soft tokens. "
                    "If set (with lclm=true), it is wrapped as the memory block and "
                    "the prompt/question stays verbatim. Ignored when lclm=false.",
    )


class GenerateResponse(BaseModel):
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    gen_time_s: float
    tokens_per_s: float
    eff_bits: float
    kv_bytes: int
    fp16_kv_bytes: int
    kv_compression_x: Optional[float] = None


@app.get("/health")
async def health():
    """Liveness, plus per-worker readiness in the shape callers already parse.

    The `*_ready` flags used to mean "a handle object exists", which is true the
    instant the app starts and stays true while the model is still loading --
    so the one question the field was asked was the one it could not answer.
    They now mean the worker's warmup returned.
    """
    states = LIFECYCLE.readiness() if LIFECYCLE is not None else {"components": []}
    ready = {c["name"]: c["ready"] for c in states["components"]}
    return {
        "status": "ok",  # the process is serving; see /ready for what it can serve
        "compressor_ready": ready.get(COMPRESSOR, False),
        "attentionrag_ready": ready.get(ATTENTIONRAG, False),
        "turboquant_ready": ready.get(TURBOQUANT, False),
        "lclm_ready": ready.get(LCLM, False),
    }


@app.get("/ready")
async def ready(response: Response):
    """Every component, its state, and whether anything is wrong.

    Two separate facts, deliberately. `ready` is whether the service can do its
    job at all, and only the critical workers decide it. `degraded` is whether
    anything is missing, optional workers included. Collapsing them would either
    page somebody for a service that is working or hide a dead worker behind a
    green light.

    503 when a critical worker is missing, so a load balancer takes this replica
    out; 200 while merely degraded, because the routes that still work should
    still get traffic.
    """
    if LIFECYCLE is None:
        response.status_code = 503
        return {"ready": False, "degraded": True, "components": [],
                "detail": "the service is still starting"}
    states = LIFECYCLE.readiness()
    if not states["ready"]:
        response.status_code = 503
    return states


@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    """Compress text with LLMLingua-2, and -- when `question` is provided -- run
    AttentionRAG in parallel and MERGE the two keep-decisions token-by-token over
    the original text (intersection or union). Without `question`, behaves as the
    original LLMLingua-only endpoint.
    """
    if req.mode not in ("intersection", "union"):
        raise HTTPException(status_code=422, detail=f"bad mode: {req.mode!r}")

    merging = bool(req.question and req.question.strip())
    # LLMLingua is the canonical spine for the merge, so we need its word labels.
    need_labels = req.return_labels or merging

    compressor = require(COMPRESSOR)
    # Asked for before anything runs, not discovered halfway through. A caller
    # who asked for question-aware compression is told the service cannot do
    # that right now, rather than being handed LLMLingua-only output for a
    # request that named a second method.
    attn_service = require(ATTENTIONRAG) if merging else None

    llm_coro = compressor.compress.remote.aio(
        req.text, rate=req.rate, return_labels=need_labels
    )

    if not merging:
        try:
            out = await llm_coro
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Modal call failed: {exc}")
        return CompressResponse(
            compressed_prompt=out["compressed_prompt"],
            origin_tokens=out["origin_tokens"],
            compressed_tokens=out["compressed_tokens"],
            rate=_parse_rate(out["rate"]),
            ratio=out["ratio"],
            word_labels=[
                [w, l] for w, l in normalize_labels(
                    out.get("fn_labeled_original_prompt") or out.get("word_labels")
                )
            ] if req.return_labels else None,
            merged=False,
        )

    # --- parallel: LLMLingua + AttentionRAG -------------------------------
    attn_coro = attn_service.compress_spans.remote.aio(
        req.text,
        req.question,
        chunk_size=req.chunk_size,
        top_k=req.top_k,
        use_openai_hint=req.use_openai_hint,
    )
    llm_out, attn_out = await asyncio.gather(
        llm_coro, attn_coro, return_exceptions=True
    )

    if isinstance(llm_out, Exception):
        raise HTTPException(status_code=502, detail=f"LLMLingua failed: {llm_out}")
    word_labels = llm_out.get("fn_labeled_original_prompt") or llm_out.get("word_labels")
    if not word_labels:
        raise HTTPException(status_code=502, detail="LLMLingua returned no labels")

    # AttentionRAG failure -> treat as empty -> fallback to LLMLingua-only.
    if isinstance(attn_out, Exception):
        kept_spans, attn_empty = [], True
        hint, kept_chunks = None, "0/0 (attnrag failed)"
    else:
        kept_spans = attn_out.get("kept_spans", [])
        # A "none" hint prefix normally makes us fall back to LLMLingua-only. But
        # for a single-chunk (short, <= chunk_size) input, compress_spans keeps
        # the whole chunk, so we honor those spans instead of discarding them on
        # a none hint. Genuine emptiness (no kept_spans) still falls back, via
        # merge_compress's `not kept_spans` guard.
        single_chunk = attn_out.get("n_chunks", 0) == 1
        attn_empty = attn_out.get("is_empty_prefix", False) and not single_chunk
        hint = attn_out.get("hint_prefix")
        kept_chunks = f"{attn_out.get('n_kept_chunks', 0)}/{attn_out.get('n_chunks', 0)}"

    merged = merge_compress(
        req.text, word_labels, kept_spans, mode=req.mode, attnrag_empty=attn_empty
    )
    ratio = merged["n_words"] / max(merged["n_kept"], 1)
    return CompressResponse(
        compressed_prompt=merged["compressed_prompt"],
        origin_tokens=llm_out["origin_tokens"],
        compressed_tokens=llm_out["compressed_tokens"],
        rate=_parse_rate(llm_out["rate"]),
        ratio=llm_out["ratio"],
        word_labels=merged["word_labels"],
        mode=merged["mode"],
        merged=True,
        used_llmlingua_fallback=merged["used_llmlingua_fallback"],
        words_total=merged["n_words"],
        words_kept=merged["n_kept"],
        merged_ratio=f"{ratio:.2f}x",
        attentionrag_hint_prefix=hint,
        attentionrag_kept_chunks=kept_chunks,
    )


@app.post("/compress_rag")
async def compress_rag(req: RagRequest):
    """Two-stage, question-aware compression: reranker coarse + LLMLingua-2 tokens."""
    compressor = require(COMPRESSOR)
    try:
        out = await compressor.compress_rag.remote.aio(
            req.instruction,
            req.question,
            req.documents,
            rate=req.rate,
            target_token=req.target_token,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
        )
    except Exception as exc:  # surface Modal errors as a clean 502
        raise HTTPException(status_code=502, detail=f"Modal call failed: {exc}")

    return out


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text with a TurboQuant-compressed KV cache on a warm A100 worker.
    The model is already loaded (warmed at server startup), so there is no
    model-startup cost on this request path.

    Routing: `lclm=true` -> LCLM (context compressed to latent soft tokens) +
    TurboQuant on the decoder KV cache; otherwise the default Qwen TurboQuant
    route (unchanged). Both return the same response shape."""
    worker = require(LCLM if req.lclm else TURBOQUANT)
    try:
        if req.lclm:
            out = await worker.generate.remote.aio(
                req.prompt,
                bit_width=req.bit_width,
                max_new_tokens=req.max_new_tokens,
                outlier_channels=req.outlier_channels,
                outlier_bits=req.outlier_bits,
                context=req.context,
            )
        else:
            out = await worker.generate.remote.aio(
                req.prompt,
                bit_width=req.bit_width,
                max_new_tokens=req.max_new_tokens,
                outlier_channels=req.outlier_channels,
                outlier_bits=req.outlier_bits,
            )
    except Exception as exc:  # surface Modal errors as a clean 502
        raise HTTPException(status_code=502, detail=f"Modal call failed: {exc}")
    return GenerateResponse(**out)


# --------------------------------------------------------------------------- #
# Unified playground: Layer 1 (compression) -> Layer 2 (downstream LLM)
# --------------------------------------------------------------------------- #
class PlaygroundRequest(BaseModel):
    text: str = Field(..., description="The input text to compress + run through an LLM")
    question: Optional[str] = Field(
        None, description="Question/instruction asked of the downstream LLM (always applies)"
    )
    attn_query: Optional[str] = Field(
        None, description="AttentionRAG focus query (per-panel); falls back to `question`"
    )
    # ---- Layer 1: context compression --------------------------------------
    methods: List[str] = Field(
        default_factory=lambda: ["llmlingua"],
        description="Subset of ['llmlingua','attentionrag']. Both -> token-merge.",
    )
    combine: str = Field("intersection", description="'intersection' or 'union' when both methods run")
    rate: float = Field(0.7, gt=0, le=1, description="LLMLingua keep-rate")
    # ---- Layer 2: downstream LLM -------------------------------------------
    backend: str = Field("claude", description="'claude' | 'chatgpt' | 'qwen' | 'lclm'")
    model: Optional[str] = Field(None, description="Model id for the black-box backends")
    quantized: bool = Field(True, description="Qwen/LCLM: use the quantized KV cache (4-bit) vs 8-bit")
    max_new_tokens: int = Field(4096, ge=1, le=16384, description="Output budget (effectively uncapped)")
    instruction: Optional[str] = Field(None, description="System/instruction for the LLM")


def _splice_spans(text: str, spans) -> str:
    """AttentionRAG-only reconstruction: stitch kept char-spans of the original."""
    spans = sorted((s, e) for s, e in (spans or []) if e > s)
    if not spans:
        return text  # nothing kept -> don't wipe the input
    return " ".join(text[s:e].strip() for s, e in spans).strip()


def _word_tokens(s: str) -> int:
    return len((s or "").split())


def _with_token_stats(out: dict, original: str) -> dict:
    """Normalize any Layer-1 result to a uniform {origin_tokens, compressed_tokens, ratio}.

    The per-method branches below report counts under different keys (LLMLingua:
    origin_tokens/compressed_tokens; merge: n_words/n_kept; AttentionRAG/passthrough:
    none at all). The /playground frontend reads ONE shape, so collapse them here.
    """
    origin = out.get("origin_tokens")
    if origin is None:
        origin = out.get("n_words")
    if origin is None:
        origin = _word_tokens(original)
    kept = out.get("compressed_tokens")
    if kept is None:
        kept = out.get("n_kept")
    if kept is None:
        kept = _word_tokens(out.get("compressed_text", ""))
    out["origin_tokens"] = origin
    out["compressed_tokens"] = kept
    out["ratio"] = round(origin / kept, 2) if kept else None
    return out


async def _layer1(req: PlaygroundRequest) -> dict:
    """Run Layer-1 compression and return a token-stat-normalized result."""
    return _with_token_stats(await _layer1_raw(req), req.text)


async def _layer1_raw(req: PlaygroundRequest) -> dict:
    """Run the selected Layer-1 compressor(s); return compressed text + stats."""
    methods = {m.strip().lower() for m in req.methods}
    use_llm, use_attn = "llmlingua" in methods, "attentionrag" in methods
    # AttentionRAG uses its dedicated per-panel query, falling back to the LLM question.
    q = (req.attn_query or req.question or "").strip()

    if not use_llm and not use_attn:  # passthrough
        return {"compressed_text": req.text, "methods": [], "note": "no compression"}

    # AttentionRAG needs a query; without one, drop it (fall back to LLMLingua).
    if use_attn and not q:
        use_attn = False
        attn_note = "attentionrag skipped (no question)"
    else:
        attn_note = None

    if use_llm and use_attn:
        compressor = require(COMPRESSOR)
        attn_service = require(ATTENTIONRAG)
        llm_coro = compressor.compress.remote.aio(req.text, rate=req.rate, return_labels=True)
        attn_coro = attn_service.compress_spans.remote.aio(req.text, q)
        llm_out, attn_out = await asyncio.gather(llm_coro, attn_coro, return_exceptions=True)
        if isinstance(llm_out, Exception):
            raise HTTPException(status_code=502, detail=f"LLMLingua failed: {llm_out}")
        labels = llm_out.get("fn_labeled_original_prompt") or llm_out.get("word_labels")
        kept_spans = [] if isinstance(attn_out, Exception) else attn_out.get("kept_spans", [])
        attn_empty = isinstance(attn_out, Exception) or not kept_spans
        merged = merge_compress(req.text, labels, kept_spans, mode=req.combine, attnrag_empty=attn_empty)
        return {
            "compressed_text": merged["compressed_prompt"], "methods": ["llmlingua", "attentionrag"],
            "combine": req.combine, "word_labels": merged["word_labels"],
            "n_words": merged["n_words"], "n_kept": merged["n_kept"],
            "used_llmlingua_fallback": merged["used_llmlingua_fallback"],
        }

    if use_llm:
        out = await require(COMPRESSOR).compress.remote.aio(
            req.text, rate=req.rate, return_labels=True)
        return {"compressed_text": out["compressed_prompt"], "methods": ["llmlingua"],
                "origin_tokens": out["origin_tokens"], "compressed_tokens": out["compressed_tokens"],
                "note": attn_note}

    # AttentionRAG only
    attn_out = await require(ATTENTIONRAG).compress_spans.remote.aio(req.text, q)
    return {"compressed_text": _splice_spans(req.text, attn_out.get("kept_spans", [])),
            "methods": ["attentionrag"],
            "kept_chunks": f"{attn_out.get('n_kept_chunks', 0)}/{attn_out.get('n_chunks', 0)}"}


async def _layer2(req: PlaygroundRequest, context: str) -> dict:
    """Run the compressed context through the selected downstream LLM."""
    backend = req.backend.strip().lower()
    q = (req.question or "").strip()
    instruction = req.instruction or "Answer using the provided context."

    if backend in ("claude", "anthropic", "chatgpt", "openai", "gpt"):
        provider = _canonical_provider(backend)
        key = _key_for(provider)
        if not key:
            raise HTTPException(status_code=400, detail=f"No API key for {provider}.")
        model = req.model or BLACKBOX_DEFAULT_MODEL[provider]
        user = f"{context}\n\nQuestion: {q}" if q else context
        caller = _call_claude if provider == "claude" else _call_openai
        # SDK calls are blocking -> offload so we don't stall the event loop.
        text, in_tok, out_tok = await asyncio.to_thread(
            caller, model, instruction, [{"role": "user", "content": user}],
            req.max_new_tokens, 0.7, key,
        )
        return {"backend": provider, "model": model, "text": text,
                "input_tokens": in_tok, "output_tokens": out_tok}

    bit_width = 4 if req.quantized else 8
    if backend == "qwen":
        prompt = f"{context}\n\nQuestion: {q}" if q else context
        out = await require(TURBOQUANT).generate.remote.aio(
            prompt, bit_width=bit_width, max_new_tokens=req.max_new_tokens)
        return {"backend": "qwen", "quantized": req.quantized, **out}
    if backend == "lclm":
        out = await require(LCLM).generate.remote.aio(
            q or "Summarize the key facts in the context.",
            context=context, bit_width=bit_width, max_new_tokens=req.max_new_tokens)
        return {"backend": "lclm", "quantized": req.quantized, **out}

    raise HTTPException(status_code=422, detail=f"unknown backend: {backend!r}")


@app.post("/playground")
async def playground(req: PlaygroundRequest):
    """End-to-end: Layer 1 (LLMLingua / AttentionRAG / both) -> Layer 2 (Claude /
    ChatGPT / Qwen / LCLM). The frontend calls this once per comparison panel."""
    if req.combine not in ("intersection", "union"):
        raise HTTPException(status_code=422, detail=f"bad combine: {req.combine!r}")
    try:
        t0 = time.perf_counter()
        layer1 = await _layer1(req)
        compress_dt = time.perf_counter() - t0
        # Uniform HARD-compression metric (Layer 1 token reduction), word-based.
        ow = len(req.text.split())
        kw = len(layer1["compressed_text"].split())
        layer1["origin_words"] = ow
        layer1["kept_words"] = kw
        layer1["hard_ratio"] = round(ow / max(kw, 1), 2)
        # Omit the compress timer entirely when there is NO hard compression.
        if layer1.get("methods"):
            layer1["compress_time_s"] = round(compress_dt, 3)
        t1 = time.perf_counter()
        layer2 = await _layer2(req, layer1["compressed_text"])
        layer2["llm_time_s"] = round(time.perf_counter() - t1, 3)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"playground failed: {exc}")
    return {"layer1": layer1, "layer2": layer2}
