"""Stability: determinism, repeat-run variance, and behaviour at the edges.

A single run is not a result. This checks that the two models in the pipeline
return the same answer when asked the same question twice, and then pushes the
compressor at inputs a live transcript really does produce - an empty pause, a
one-word utterance, a recogniser stuck in a repetition loop, and a transcript
longer than the encoder's window.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHISPER = os.environ.get("WHISPER_CLI") or shutil.which("whisper-cli") or "whisper-cli"
fails = []


def chk(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'   ' + detail if detail else ''}")
    if not ok:
        fails.append(name)
    return ok


def asr_determinism(n=5):
    print("\n[1] ASR determinism: same audio, same flags, repeated")
    for cond in ("clean", "snr0", "snr-5"):
        wav = ROOT / "audio" / f"61-70968-0000.{cond}.wav"
        outs = []
        for _ in range(n):
            p = subprocess.run([WHISPER, "-m", str(ROOT/"models/ggml-base.en.bin"),
                                "-f", str(wav), "-l", "en", "-nt", "-np", "-t", "4",
                                "-bs", "5", "-bo", "5"], capture_output=True, text=True)
            if p.returncode == 0:
                outs.append(" ".join(p.stdout.split()))
        uniq = set(outs)
        chk(f"whisper base.en is deterministic on {cond}", len(uniq) == 1 and outs,
            f"{len(outs)} runs, {len(uniq)} distinct output(s)")


def compressor_determinism(comp, C, n=5):
    print("\n[2] Compressor determinism on MPS: same text, same rate, repeated")
    asr = json.loads((ROOT / "out" / "asr.json").read_text())
    txt = C.flatten(asr["base|61-70968-0000|clean"]["hyp"])
    for rate in (0.5, 0.33, 0.2):
        outs = {C.compress(comp, txt, rate=rate)[0] for _ in range(n)}
        chk(f"LLMLingua-2 deterministic at rate {rate}", len(outs) == 1,
            f"{n} runs, {len(outs)} distinct output(s)")


def edges(comp, C):
    print("\n[3] Edge inputs a live transcript actually produces")
    cases = {
        "empty (a pause with no speech)": "",
        "whitespace only": "   ",
        "single word utterance": "OKAY",
        "two words": "RIGHT SO",
        "punctuation only": "... ,,, ?!",
        "digits only (a spoken figure)": "1947 2026 314",
        "repetition loop (recogniser stuck)": " ".join(["THANK YOU"] * 120),
    }
    for name, txt in cases.items():
        try:
            out, _ = C.compress(comp, txt, rate=0.33)
            chk(f"survives: {name}", True, f"-> {len(out.split())} words {out[:40]!r}")
        except Exception as e:
            chk(f"survives: {name}", False, f"{type(e).__name__}: {e}")

    # Long-input behaviour: XLM-R's window is 512 tokens. If the compressor
    # silently drops everything past the window, a long transcript loses its TAIL
    # entirely - and the caller is never told.
    print("\n[4] Long input: does the tail survive the encoder window?")
    corpus = json.loads((ROOT / "out" / "corpus.json").read_text())
    base = " ".join(corpus["splits"]["eval"][0]["ref_words"])
    for mult in (1, 2, 4, 8):
        txt = " ".join([base] * mult)
        marker = "ZEBRAFISH QUASAR MARGRAVE"   # unique sentinel, last words in
        txt = txt + " " + marker
        out, _ = C.compress(comp, txt, rate=0.5)
        n_in, n_out = len(txt.split()), len(out.split())
        tail_kept = any(w in out.upper() for w in marker.split())
        head_kept = base.split()[1].upper() in out.upper()
        print(f"    x{mult:<2} in={n_in:5d} out={n_out:5d} "
              f"achieved_rate={n_out/n_in:.3f}  head_kept={head_kept}  "
              f"TAIL_kept={tail_kept}")


def main():
    import compress_lib as C
    asr_determinism()
    comp, dev = C.load_compressor()
    print(f"(compressor device: {dev})")
    compressor_determinism(comp, C)
    edges(comp, C)
    print(f"\n{'STABILITY CHECKS PASS' if not fails else 'FAILURES: ' + str(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
