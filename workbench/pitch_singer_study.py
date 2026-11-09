"""Study S3: on a show with only split vocals, can pitch tell a low voice's lines from a high voice's?

Truth (per lyric line): whichever vocal mic owns the 2-5 kHz presence range during the line
(eq_study's ownership), by a clear margin over the other vocal mics; unclear lines are skipped.
Each singer's range comes from a label (as Stefan would set it in the People Picker), given here
by --high NAME[,NAME]. Test: the median pitch (timbre.yin) of the split vocals over the line,
above --split-hz means "high". Prints scores only, never words.

  /usr/bin/python3 workbench/pitch_singer_study.py --stems DIR --analysis DIR --lyrics DIR
      --parts CACHE_DIR --high TRIANNA
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music_detector.core import timbre as T                         # noqa: E402
from eq_study import band_energy, PRESENCE, RATE, VOX, SKIP, decode  # noqa: E402

MARGIN_DB = 3.0
SPLIT_CENTS = 300


def line_pitch(x, rate, a, b):
    seg = T._highpass(x[int(a * rate):int(b * rate)], rate, 100.0)
    if len(seg) < T.FRAME * 2:
        return None
    f0, clar = T.yin(T._frames(seg), rate)
    rms = np.sqrt(np.mean(T._frames(seg) ** 2, axis=1))
    ok = ~np.isnan(f0) & (rms > np.percentile(rms, 40))
    return float(np.median(f0[ok])) if ok.sum() >= 5 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--parts", required=True, help="horns_eval cache dir with <song>/parts/")
    ap.add_argument("--high", required=True)
    a = ap.parse_args()
    import soundfile as sf
    high = {h.strip().upper() for h in a.high.split(",")}
    vox = [w for w in sorted(glob.glob(os.path.join(a.stems, "*.wav")))
           if VOX.search(os.path.basename(w)) and not SKIP.search(os.path.basename(w))]
    inst = [w for w in sorted(glob.glob(os.path.join(a.stems, "*.wav")))
            if w not in vox and not SKIP.search(os.path.basename(w))]
    who = {w: re.sub(r"^\d+\s*|\bVOX\b|\.wav$", "", os.path.basename(w), flags=re.I).strip().upper()
           for w in vox}
    rows = []
    for jp in sorted(glob.glob(os.path.join(a.analysis, "whoplays_*.json"))):
        js = json.load(open(jp))
        label = js["song"]
        lp = os.path.join(a.lyrics, f"lyrics_{label}.json")
        key = re.sub(r"[^A-Za-z0-9]+", "_", label)
        pp = os.path.join(a.parts, key, "parts", "vocals.flac")
        man = os.path.join(a.parts, key, "parts", ".separate.json")
        if js.get("stem_offset_s") is None or not (os.path.isfile(lp) and os.path.isfile(pp)):
            continue
        so, L = float(js["stem_offset_s"]), float(js["song_len_s"])
        pad = json.load(open(man)).get("pad_s", 1.0) if os.path.isfile(man) else 1.0
        v, sr = sf.read(pp)
        v = v[int(pad * sr):]
        ib = None
        for w in inst:
            e, _ = band_energy(decode(w, so, L, RATE), *PRESENCE)
            ib = e if ib is None else ib[:len(e)] + e[:len(ib)]
        ratio = {}
        mics = {}
        for w in vox:
            x = decode(w, so, L, RATE)
            pres, tot = band_energy(x, *PRESENCE)
            m = min(len(pres), len(ib))
            ratio[w] = 10 * np.log10(pres[:m] + 1e-12) - 10 * np.log10(ib[:m] + 1e-12)
            mics[w] = x
        for l in json.load(open(lp)).get("lines") or []:
            s0, s1 = float(l["start"]), float(l["end"])
            if s1 - s0 < 0.6:
                continue
            b0, b1 = int(s0 / 0.5), max(int(s0 / 0.5) + 1, int(np.ceil(s1 / 0.5)))
            score = {w: float(np.mean(r[b0:b1])) - float(np.percentile(r, 20))
                     for w, r in ratio.items() if b1 <= len(r)}
            if not score:
                continue
            best = sorted(score.items(), key=lambda kv: -kv[1])
            if best[0][1] < 6 or (len(best) > 1 and best[0][1] - best[1][1] < MARGIN_DB):
                continue                                   # nobody clearly owns it / a duet
            singer = who[best[0][0]]
            own_f0 = line_pitch(mics[best[0][0]], RATE, s0, s1)
            split_f0 = line_pitch(v, sr, s0, s1)
            rows.append({"song": label, "singer": singer, "high": singer in high,
                         "own_f0": own_f0, "split_f0": split_f0})
    print(f"lines with a clear singer: {len(rows)}")
    for s in sorted({r["singer"] for r in rows}):
        f = [r["own_f0"] for r in rows if r["singer"] == s and r["own_f0"]]
        g = [r["split_f0"] for r in rows if r["singer"] == s and r["split_f0"]]
        print(f"  {s:10} {'high' if s in high else 'low '} lines {sum(r['singer'] == s for r in rows):4}"
              f"  own-mic pitch {np.median(f) if f else float('nan'):6.0f} Hz"
              f"  split pitch {np.median(g) if g else float('nan'):6.0f} Hz")
    ok = [r for r in rows if r["split_f0"]]
    for hz in (180, 200, 220, 240, 260):
        acc = np.mean([(r["split_f0"] > hz) == r["high"] for r in ok]) if ok else float("nan")
        hi = [r for r in ok if r["high"]]
        lo = [r for r in ok if not r["high"]]
        hr = np.mean([r["split_f0"] > hz for r in hi]) if hi else float("nan")
        lr = np.mean([r["split_f0"] <= hz for r in lo]) if lo else float("nan")
        print(f"split at {hz} Hz: right {acc:.0%}  (high voice right {hr:.0%}, low voices right {lr:.0%})")
    # relative: in each song, split its lines' pitches into two groups (deterministic 1-D 2-means);
    # the higher group is the high voice -- only when the groups are clearly apart
    right = tot = unsure = 0
    for song in sorted({r["song"] for r in ok}):
        f = np.array([r["split_f0"] for r in ok if r["song"] == song])
        hi_true = np.array([r["high"] for r in ok if r["song"] == song])
        lg = np.log2(f)
        c0, c1 = np.percentile(lg, 25), np.percentile(lg, 75)
        for _ in range(20):
            g = np.abs(lg - c1) < np.abs(lg - c0)
            if g.all() or (~g).all():
                break
            c0, c1 = lg[~g].mean(), lg[g].mean()
        apart = (c1 - c0) * 1200 >= SPLIT_CENTS
        if not apart:
            unsure += len(f)                    # one voice range, or too close: the lead gets them
            continue
        pred = np.abs(lg - c1) < np.abs(lg - c0)
        right += int((pred == hi_true).sum())
        tot += len(f)
    print(f"per-song two groups (>= {SPLIT_CENTS} cents apart): right {right}/{tot}"
          f" = {right / max(1, tot):.0%}; {unsure} lines in songs with one range (go to the lead)")
    near = [r for r in ok if 190 <= r["split_f0"] <= 250]
    print(f"lines in the overlap zone 190-250 Hz: {len(near)} of {len(ok)}")


if __name__ == "__main__":
    main()
