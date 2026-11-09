"""Proof run: can we tell when the horns play from split parts alone?

For each song of a show WITH real horn mics:
  1. mix the band's real stems for the song (as a board mix would be),
  2. split that mix with htdemucs_6s (cached under --cache),
  3. truth  = any horn mic playing (its own activity, as whoplays measures it),
     baseline = "the other part is loud",
     new      = loud AND horn-like (timbre.horn_likeness), in the other / guitar / keys parts.
Scores are per 0.5 s bin. Read-only on the show; everything written goes under --cache.

  /usr/bin/python3 workbench/horns_eval.py --stems DIR --analysis DIR --cache DIR [--songs 1,2]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_detector import separate as S                      # noqa: E402
from music_detector.core import detectors as D, timbre as T, whoplays as W   # noqa: E402

HORN = re.compile(r"\b(SAX|TRUMPET|TPT|HORN|TROMBONE|TBN)\b", re.I)
SKIP = re.compile(r"\b(DJ|IPOD|LINE ?[LR]|LINEL|LINER)\b", re.I)
NOT_BAND = re.compile(r"\b(VOX|VOCAL|TOM)\b", re.I)   # singers move with the horns (fills
#   between lines), so they'd fake horn entrances; toms are rare and loud
BIN = T.BIN_S
PART_ON = 0.30            # a part "plays" at this fraction of its loud level in the song


def decode(path, start, dur, rate=T.RATE):
    p = subprocess.run([W.FFMPEG, "-v", "error", "-ss", f"{max(0, start):.3f}", "-t", f"{dur:.3f}",
                        "-i", path, "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    x = np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)
    if start < 0:
        x = np.concatenate([np.zeros(int(-start * rate)), x])
    return x


def mix_song(stems, start, dur, out):
    if os.path.isfile(out):
        return out
    acc = None
    for f in stems:
        x = decode(f, start, dur)
        acc = x if acc is None else acc[:len(x)] + x[:len(acc)]
    acc = acc / (np.max(np.abs(acc)) + 1e-9) * 0.9
    import soundfile as sf
    sf.write(out, acc.astype(np.float32), T.RATE)
    return out


def part_act(x, nb):
    per = int(BIN * T.RATE)
    r = np.array([np.sqrt(np.mean(x[i * per:(i + 1) * per] ** 2)) if i * per < len(x) else 0.0
                  for i in range(nb)])
    return r / (np.percentile(r, 92) + 1e-9)


TRUTH_HF = (1000, 4000)   # a horn right on its mic owns this range of that mic; bleed doesn't
TRUTH_ABOVE_DB = 10.0     # playing = this far above the mic's own resting (20th pct) ratio


def _hf(x, rate=T.RATE, bin_s=BIN):
    per = int(bin_s * rate)
    n = len(x) // per
    X = x[:n * per].reshape(n, per)
    S = np.abs(np.fft.rfft(X * np.hanning(per), axis=1)) ** 2
    f = np.fft.rfftfreq(per, 1.0 / rate)
    return S[:, (f >= TRUTH_HF[0]) & (f <= TRUTH_HF[1])].sum(axis=1)


def truth_bins(horn_files, band_files, so, L, nb):
    """Horns playing per bin: a horn mic's 1-4 kHz energy against the band's (the instrument mics, not vocals),
    well above that mic's resting level of bleed. Level alone can't tell -- these mics hear the
    whole band all song."""
    band = None
    for f in band_files:
        h = _hf(decode(f, so, L))
        band = h if band is None else band[:len(h)] + h[:len(band)]
    on = np.zeros(nb, dtype=bool)
    for f in horn_files:
        h = _hf(decode(f, so, L))
        m = min(len(h), len(band), nb)
        r = 10 * np.log10(h[:m] + 1e-12) - 10 * np.log10(band[:m] + 1e-12)
        if np.max(h[:m]) < 1e-9 or np.percentile(h[:m], 95) < 1e-4 * np.percentile(band[:m], 95):
            continue                          # mic off / unplugged for this song
        on[:m] |= r > np.percentile(r, 20) + TRUTH_ABOVE_DB
    return on


def f1(pred, truth):
    tp = float(np.sum(pred & truth))
    p = tp / max(1, pred.sum())
    r = tp / max(1, truth.sum())
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def run_song(js, stems, horn_files, cache):
    so, L = float(js["stem_offset_s"]), float(js["song_len_s"])
    name = js["song"]
    key = re.sub(r"[^A-Za-z0-9]+", "_", name)
    d = os.path.join(cache, key)
    os.makedirs(d, exist_ok=True)
    mix = mix_song(stems, so - 1.0, L + 2.0, os.path.join(d, "mix.wav"))
    res = S.split_song(mix, 1.0, 1.0 + L, os.path.join(d, "parts"))
    if not res["ok"]:
        raise RuntimeError(res["error"])
    pad = res["pad_s"]
    nb = int(L / BIN)
    import soundfile as sf
    parts = {}
    for p in ("other", "guitar", "keys"):
        x, sr = sf.read(os.path.join(d, "parts", f"{p}.flac"))
        x = x[int(pad * sr):int((pad + L) * sr)]
        m = T.measure(x, sr, n_bins=nb)
        parts[p] = {"act": part_act(x, nb), "m": m, "h": T.horn_likeness(m)}
    truth = truth_bins(horn_files, [f for f in stems if f not in horn_files
                                    and not NOT_BAND.search(os.path.basename(f))], so, L, nb)
    return name, truth, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--songs", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    wavs = sorted(glob.glob(os.path.join(a.stems, "*.wav")))
    stems = [w for w in wavs if not SKIP.search(os.path.basename(w))]
    horns = [w for w in stems if HORN.search(os.path.basename(w))]
    want = {s.strip() for s in a.songs.split(",") if s.strip()}
    rows = []
    for jp in sorted(glob.glob(os.path.join(a.analysis, "whoplays_*.json"))):
        js = json.load(open(jp))
        num = js["song"].split(" - ")[0]
        if want and num not in want:
            continue
        if js.get("stem_offset_s") is None:
            continue
        name, truth, parts = run_song(js, stems, horns, a.cache)
        rows.append({"song": name, "truth": truth.tolist(),
                     "parts": {p: {"act": v["act"].tolist(), "h": v["h"].tolist(),
                                   "m": {k: np.nan_to_num(x).tolist() for k, x in v["m"].items()}}
                               for p, v in parts.items()}})
        base = parts["other"]["act"] >= PART_ON
        print(f"{name}: horns {truth.mean():.0%} of song, baseline F1 {f1(base, truth)[2]:.2f}",
              flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
