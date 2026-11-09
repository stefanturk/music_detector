"""Study S2: does "this mic owns its frequency range" tell singing from bleed better than loudness?

Truth: the song's lyric lines (Stefan's corrected ones where he fixed them, else Whisper's), as
"someone is singing" per 0.5 s bin. Read-only on the show; prints scores only, never words.

  loudness: the vocal mics' summed envelope over its show-wide 92nd percentile, on above 0.32
            (today's whoplays rule, without its hysteresis)
  eq      : any vocal mic's 2-5 kHz energy against the instrument mics' 2-5 kHz energy, well above
            that mic's own resting ratio (its 20th percentile) -- the voice owns the presence range
            on its own mic; bleed doesn't

  /usr/bin/python3 workbench/eq_study.py --stems DIR --analysis DIR --lyrics DIR
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from horns_eval import decode, BIN                                  # noqa: E402

VOX = re.compile(r"\b(VOX|VOCAL|VOC)\b", re.I)
SKIP = re.compile(r"\b(DJ|IPOD|LINE ?[LR]|LINEL|LINER)\b", re.I)
PRESENCE = (2000, 5000)
RATE = 16000


def band_energy(x, lo, hi, rate=RATE, bin_s=BIN):
    per = int(bin_s * rate)
    n = len(x) // per
    X = x[:n * per].reshape(n, per)
    S = np.abs(np.fft.rfft(X * np.hanning(per), axis=1)) ** 2
    f = np.fft.rfftfreq(per, 1.0 / rate)
    return S[:, (f >= lo) & (f <= hi)].sum(axis=1), S.sum(axis=1)


def truth_of(lines, nb):
    t = np.zeros(nb, dtype=bool)
    for l in lines:
        a, b = int(float(l["start"]) / BIN), int(np.ceil(float(l["end"]) / BIN))
        t[max(0, a):max(0, min(nb, b))] = True
    return t


def f1(p, t):
    tp = float((p & t).sum())
    P, R = tp / max(1, p.sum()), tp / max(1, t.sum())
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--above", type=float, default=10.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    wavs = [w for w in sorted(glob.glob(os.path.join(a.stems, "*.wav")))
            if not SKIP.search(os.path.basename(w))]
    vox = [w for w in wavs if VOX.search(os.path.basename(w))]
    inst = [w for w in wavs if w not in vox]
    rows = []
    for jp in sorted(glob.glob(os.path.join(a.analysis, "whoplays_*.json"))):
        js = json.load(open(jp))
        label = js["song"]
        lp = os.path.join(a.lyrics, f"lyrics_{label}.json")
        if js.get("stem_offset_s") is None or not os.path.isfile(lp):
            continue
        ly = json.load(open(lp))
        H = ly.get("heard_lines")
        corrected = H is not None and [l.get("line") for l in ly["lines"]] != [l.get("line") for l in H]
        so, L = float(js["stem_offset_s"]), float(js["song_len_s"])
        nb = int(L / BIN)
        t = truth_of(ly.get("lines") or [], nb)
        ib = None
        for w in inst:
            e, _ = band_energy(decode(w, so, L, RATE), *PRESENCE)
            ib = e if ib is None else ib[:len(e)] + e[:len(ib)]
        loud = np.zeros(nb)
        eq = np.zeros(nb, dtype=bool)
        for w in vox:
            x = decode(w, so, L, RATE)
            pres, tot = band_energy(x, *PRESENCE)
            m = min(nb, len(pres), len(ib))
            if np.percentile(tot[:m], 95) < 1e-8:
                continue                                   # mic off this song
            loud[:m] += tot[:m]
            r = 10 * np.log10(pres[:m] + 1e-12) - 10 * np.log10(ib[:m] + 1e-12)
            eq[:m] |= r > np.percentile(r, 20) + a.above
        lv = np.sqrt(loud)
        loud_on = lv / (np.percentile(lv, 92) + 1e-12) > 0.32
        rows.append({"song": label, "corrected": corrected, "truth": t.tolist(),
                     "loud": loud_on.tolist(), "eq": eq.tolist()})
        print(f"{label:<36} {'corrected' if corrected else 'whisper  '} sung {t.mean():4.0%}  "
              f"loudness F1 {f1(loud_on, t)[2]:.2f}  eq F1 {f1(eq, t)[2]:.2f}", flush=True)
    T = np.concatenate([np.array(r["truth"]) for r in rows])
    for k in ("loud", "eq"):
        P = np.concatenate([np.array(r[k]) for r in rows])
        print(f"ALL {k:5} P/R/F1 " + " ".join(f"{v:.3f}" for v in f1(P, T)))
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
