"""Render each passage to 16 kHz mono WAV, clean and at controlled SNRs.

Noise is BABBLE built from LibriSpeech speakers that are in neither the TUNE nor
the EVAL split, so a voice used as interference is never a voice under test.

MEASUREMENT (this is where the first version was wrong, twice):

 1. ffmpeg's `volumedetect` only accepts s16, so it converts float input down to
    s16 before measuring and CLIPS ITS OWN MEASUREMENT. It can therefore never
    report that a signal was clipped. All levels here come from `astats`, which
    measures in float (verified: astats reads +9.46 dB peak on a float file that
    volumedetect reports as 0.0 dB).

 2. Boosting the babble to reach a low SNR pushes it past full scale, so writing
    the noise to s16 clipped it and the realised SNR undershot. Everything is
    therefore mixed in FLOAT, and the finished mix is scaled by a single uniform
    gain `u` that puts its peak at -1 dBFS. A uniform gain moves speech and noise
    together, so it cannot change the SNR; it only buys headroom.

The realised SNR is not re-derived from the arithmetic above. Both components are
re-rendered through the identical gain chain that produced the delivered file and
measured independently, and that measured difference is what lands in the manifest.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUD = ROOT / "audio"
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or "ffprobe"
SNRS = [10, 5, 0, -5]
PEAK_TARGET_DBFS = -1.0
TOL_DB = 0.05

_RMS = re.compile(r"RMS level dB:\s*(-?[\d.]+|-inf)")
_PEAK = re.compile(r"Peak level dB:\s*(-?[\d.]+|-inf)")


def sh(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:8])}... failed:\n{p.stderr[-2000:]}")
    return p


def levels(wav: Path):
    """(rms_dbfs, peak_dbfs) measured in FLOAT via astats, never volumedetect."""
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(wav), "-af",
         "astats=measure_overall=Peak_level+RMS_level:measure_perchannel=none",
         "-f", "null", "-"], capture_output=True, text=True)
    r, k = _RMS.search(p.stderr), _PEAK.search(p.stderr)
    if not r or not k:
        raise RuntimeError(f"no astats levels for {wav}:\n{p.stderr[-1500:]}")
    return float(r.group(1)), float(k.group(1))


def duration(wav: Path) -> float:
    return float(sh([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(wav)]).stdout.strip())


def concat(paths, out: Path):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p}'\n" for p in paths))
    sh([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
        "-safe", "0", "-i", str(lst), "-ac", "1", "-ar", "16000", str(out)])
    lst.unlink()
    return out


def gain_to(src: Path, out: Path, db: float, fmt: str):
    """Apply `db` gain in float precision. aformat BEFORE volume is required:
    without it the volume filter negotiates s16 and clips the gain stage."""
    sh([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-af", f"aformat=sample_fmts=fltp,volume={db:.4f}dB:precision=float",
        "-c:a", fmt, "-ac", "1", "-ar", "16000", str(out)])
    return out


def main():
    corpus = json.loads((ROOT / "out" / "corpus.json").read_text())
    AUD.mkdir(exist_ok=True)

    bed = AUD / "_babble_bed.wav"
    if not bed.exists():
        concat(corpus["babble_flacs"], bed)
    bed_rms, bed_peak = levels(bed)
    print(f"babble bed: {duration(bed):.1f}s rms={bed_rms:.2f} peak={bed_peak:.2f} dBFS "
          f"({len(corpus['babble_flacs'])} clips, speakers {corpus['proof']['babble']})")

    manifest = []
    for split in ("tune", "eval"):
        for p in corpus["splits"][split]:
            pid = p["passage_id"]
            clean = AUD / f"{pid}.clean.wav"
            if not clean.exists():
                concat(p["flacs"], clean)
            sig_rms, sig_peak = levels(clean)
            dur = duration(clean)
            manifest.append({"passage_id": pid, "split": split, "cond": "clean",
                             "snr_target": None, "wav": str(clean),
                             "dur_s": round(dur, 2), "sig_dbfs": sig_rms})

            # Unity-gain babble window for THIS passage's length. The name must not
            # collide with any `_n{snr}` file: the first version called this
            # `_noise0.wav`, which is exactly the snr=0 output name, so the snr=0
            # gain was silently skipped and "0 dB SNR" was really the raw bed.
            seg = AUD / f"{pid}._seg_unity.wav"
            if not seg.exists():
                sh([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-stream_loop", "-1", "-i", str(bed), "-af", f"atrim=0:{dur:.3f}",
                    "-ac", "1", "-ar", "16000", str(seg)])
            seg_rms, _ = levels(seg)

            for snr in SNRS:
                g = sig_rms - snr - seg_rms
                nf32 = AUD / f"{pid}._n{snr}.f32.wav"
                gain_to(seg, nf32, g, "pcm_f32le")

                mixf32 = AUD / f"{pid}._mix{snr}.f32.wav"
                sh([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(clean), "-i", str(nf32), "-filter_complex",
                    "[0:a][1:a]amix=inputs=2:duration=first:normalize=0,"
                    "aformat=sample_fmts=fltp[o]", "-map", "[o]",
                    "-c:a", "pcm_f32le", "-ac", "1", "-ar", "16000", str(mixf32)])
                _, mix_peak = levels(mixf32)
                u = min(0.0, PEAK_TARGET_DBFS - mix_peak)

                out = AUD / f"{pid}.snr{snr}.wav"
                gain_to(mixf32, out, u, "pcm_s16le")

                # Independent check: push each component through the SAME chain
                # and measure the delivered levels rather than re-deriving them.
                sc = gain_to(clean, AUD / f"{pid}._sc.wav", u, "pcm_s16le")
                nc = gain_to(nf32, AUD / f"{pid}._nc.wav", u, "pcm_s16le")
                sc_rms, sc_peak = levels(sc)
                nc_rms, nc_peak = levels(nc)
                realised = sc_rms - nc_rms
                _, out_peak = levels(out)

                manifest.append({
                    "passage_id": pid, "split": split, "cond": f"snr{snr}",
                    "snr_target": snr, "snr_realised": round(realised, 3),
                    "wav": str(out), "dur_s": round(dur, 2),
                    "sig_dbfs": round(sc_rms, 3), "noise_dbfs": round(nc_rms, 3),
                    "uniform_gain_db": round(u, 3),
                    "mix_peak_dbfs": round(out_peak, 3),
                    "component_peaks_dbfs": [round(sc_peak, 3), round(nc_peak, 3)],
                })
                for tmp in (nf32, mixf32, sc, nc):
                    tmp.unlink()
            print(f"  {pid:22s} {split:5s} {dur:6.1f}s speech_rms={sig_rms:7.2f} dBFS")

    (ROOT / "out" / "audio_manifest.json").write_text(json.dumps(manifest, indent=1))

    rows = [r for r in manifest if r.get("snr_target") is not None]
    print(f"\n{len(manifest)} renders, "
          f"{sum(r['dur_s'] for r in manifest)/60:.1f} min of audio total")
    worst = 0.0
    for snr in SNRS:
        e = [abs(r["snr_realised"] - snr) for r in rows if r["snr_target"] == snr]
        worst = max(worst, max(e))
        print(f"  target SNR {snr:+3d} dB -> max |error| {max(e):.3f} dB  (n={len(e)})")
    clipped = [r["passage_id"] for r in rows if r["mix_peak_dbfs"] > -0.5
               or max(r["component_peaks_dbfs"]) > -0.5]
    print(f"  files at/over full scale (clipping): {len(clipped)} {clipped[:3]}")
    print(f"\nworst realised-SNR error over all {len(rows)} mixes: {worst:.3f} dB")
    assert not clipped, f"clipped renders: {clipped[:5]}"
    assert worst <= TOL_DB, f"SNR calibration is wrong: worst error {worst:.3f} dB"
    print("SNR calibration verified.")


if __name__ == "__main__":
    sys.exit(main())
