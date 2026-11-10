"""Study S3b: can the COLOUR of a voice (its spectral envelope: formants) tell a show's singers apart
better than pitch, from only the split vocals? Co-Op has 4 vocal mics, so the truth exists.

Truth per lyric line: the vocal mic that owns the 2-5 kHz presence range by a clear margin (same as
pitch_singer_study). Colour per line: the mean of cepstral coefficients 2-13 (log-mel filterbank, DCT;
c0/c1 dropped so level and tilt don't count) over the line's voiced frames of the split vocals.

No training on the band. Scores (printed, never words):
  - baseline: lines the most frequent singer of each song gets (what "give it to the lead" earns)
  - per-song 2 clusters (deterministic 2-means) from colour / from pitch / from both: each cluster is
    given its majority singer; % of lines right
  - leave-one-song-out nearest centroid from the OTHER songs' labelled lines: an upper bound on the
    feature that uses labels, so it only says how separable the voices are, not what a user gets
RESULT (Co-Op 2024-11-23, 369 lines, 4 singers, 2026-11-09): voice colour does NOT beat today's rule.
"Is it the lead's line": all-to-the-lead 62%, bigger-colour-cluster 62%, pitch 53%. Colour does separate
the voices when labelled lines are known (leave-one-song-out centroid 82% vs 62% for the lead alone), but
that needs labels from the band, so it stays off. Idea left for later: a few of Stefan's own line fixes
as the voice samples.

  /usr/bin/python3 workbench/voice_colour_study.py --stems DIR --analysis DIR --lyrics DIR --parts CACHE_DIR
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
from music_detector.core import timbre as T                                # noqa: E402
from eq_study import band_energy, PRESENCE, RATE, VOX, SKIP, decode          # noqa: E402
from pitch_singer_study import line_pitch, MARGIN_DB                         # noqa: E402

NMEL, NCEP = 40, 13
FRAME_S, HOP_S = 0.032, 0.016


def _mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _imel(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _filterbank(rate, nfft):
    edges = _imel(np.linspace(_mel(100.0), _mel(min(6000.0, rate / 2 - 1)), NMEL + 2))
    bins = np.floor((nfft + 1) * edges / rate).astype(int)
    fb = np.zeros((NMEL, nfft // 2 + 1))
    for i in range(NMEL):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        for k in range(a, b):
            fb[i, k] = (k - a) / max(1, b - a)
        for k in range(b, c):
            fb[i, k] = (c - k) / max(1, c - b)
    return fb


def colour(x, rate, a, b):
    """Mean cepstral shape (c2..c12) of the line's voiced frames, or None."""
    seg = T._highpass(x[int(a * rate):int(b * rate)], rate, 100.0)
    n, hop = int(FRAME_S * rate), int(HOP_S * rate)
    if len(seg) < n * 4:
        return None
    nfft = 1 << (n - 1).bit_length()
    idx = np.arange(0, len(seg) - n, hop)
    fr = np.stack([seg[i:i + n] for i in idx]) * np.hanning(n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    keep = rms > np.percentile(rms, 50)
    if keep.sum() < 8:
        return None
    S = np.abs(np.fft.rfft(fr[keep], nfft, axis=1)) ** 2
    L = np.log(S @ _filterbank(rate, nfft).T + 1e-10)
    k = np.arange(NMEL)
    dct = np.cos(np.pi / NMEL * (k[None, :] + 0.5) * np.arange(NCEP)[:, None])
    C = L @ dct.T
    return C[:, 2:NCEP].mean(axis=0)


def kmeans2(F):
    """Deterministic 2-means: start from the two farthest-apart points."""
    d = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    c0, c1 = F[i], F[j]
    g = None
    for _ in range(30):
        g2 = ((F - c1) ** 2).sum(1) < ((F - c0) ** 2).sum(1)
        if g is not None and (g2 == g).all():
            break
        g = g2
        if g.all() or (~g).all():
            break
        c0, c1 = F[~g].mean(0), F[g].mean(0)
    return g


def majority_right(groups, truth):
    right = 0
    for gv in (True, False):
        m = groups == gv
        if m.any():
            vals, cnt = np.unique(np.array(truth)[m], return_counts=True)
            right += cnt.max()
    return int(right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--parts", required=True)
    a = ap.parse_args()
    import soundfile as sf
    vox = [w for w in sorted(glob.glob(os.path.join(a.stems, "*.wav")))
           if VOX.search(os.path.basename(w)) and not SKIP.search(os.path.basename(w))]
    inst = [w for w in sorted(glob.glob(os.path.join(a.stems, "*.wav")))
            if w not in vox and not SKIP.search(os.path.basename(w))]
    who = {w: re.sub(r"^\d+\s*|\bVOX\b|\.wav$", "", os.path.basename(w), flags=re.I).strip().upper() for w in vox}
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
        if v.ndim > 1:
            v = v.mean(axis=1)
        v = v[int(pad * sr):]
        ib = None
        for w in inst:
            e, _ = band_energy(decode(w, so, L, RATE), *PRESENCE)
            ib = e if ib is None else ib[:len(e)] + e[:len(ib)]
        ratio = {}
        for w in vox:
            pres, _ = band_energy(decode(w, so, L, RATE), *PRESENCE)
            m = min(len(pres), len(ib))
            ratio[w] = 10 * np.log10(pres[:m] + 1e-12) - 10 * np.log10(ib[:m] + 1e-12)
        for l in json.load(open(lp)).get("lines") or []:
            s0, s1 = float(l["start"]), float(l["end"])
            if s1 - s0 < 0.8:
                continue
            b0, b1 = int(s0 / 0.5), max(int(s0 / 0.5) + 1, int(np.ceil(s1 / 0.5)))
            score = {w: float(np.mean(r[b0:b1])) - float(np.percentile(r, 20)) for w, r in ratio.items() if b1 <= len(r)}
            best = sorted(score.items(), key=lambda kv: -kv[1])
            if not best or best[0][1] < 6 or (len(best) > 1 and best[0][1] - best[1][1] < MARGIN_DB):
                continue
            col = colour(v, sr, s0, s1)
            f0 = line_pitch(v, sr, s0, s1)
            if col is None or f0 is None:
                continue
            rows.append({"song": label, "singer": who[best[0][0]], "col": col, "f0": np.log2(f0) * 1200.0})
    print(f"lines with a clear singer and a usable voice: {len(rows)} in {len({r['song'] for r in rows})} songs;"
          f" singers: {sorted({r['singer'] for r in rows})}")
    tot = base = 0
    ok = {"colour": 0, "pitch": 0, "both": 0}
    songs = sorted({r["song"] for r in rows})
    allcol = np.array([r["col"] for r in rows])
    sd = allcol.std(axis=0) + 1e-9                          # one scale for every song (not per band tuned)
    usable = 0
    for s in songs:
        R = [r for r in rows if r["song"] == s]
        sg = [r["singer"] for r in R]
        if len(R) < 8 or len(set(sg)) < 2:
            continue
        usable += 1
        C = np.array([r["col"] for r in R]) / sd
        P = np.array([[r["f0"]] for r in R]) / 300.0        # 300 cents ~ one unit
        tot += len(R)
        base += max(sg.count(x) for x in set(sg))
        ok["colour"] += majority_right(kmeans2(C), sg)
        ok["pitch"] += majority_right(kmeans2(P), sg)
        ok["both"] += majority_right(kmeans2(np.hstack([C, P * 2.0])), sg)
    # the product question: the split show's lead (the most frequent singer, as a ★ would be) gets every
    # line today. Could 2 clusters (the bigger = the lead's) do better at "is this the lead's line"?
    lead = max({r["singer"] for r in rows}, key=lambda x: sum(r["singer"] == x for r in rows))
    lead_n = {"all lead": 0, "colour": 0, "pitch": 0, "both": 0}
    for s in songs:
        R = [r for r in rows if r["song"] == s]
        if len(R) < 8:
            continue
        isl = np.array([r["singer"] == lead for r in R])
        C = np.array([r["col"] for r in R]) / sd
        P = np.array([[r["f0"]] for r in R]) / 300.0
        lead_n["all lead"] += int(isl.sum())
        for name, F in (("colour", C), ("pitch", P), ("both", np.hstack([C, P * 2.0]))):
            g = kmeans2(F)
            big = g if g.sum() >= (~g).sum() else ~g          # the bigger cluster is the lead's
            lead_n[name] += int((big == isl).sum())
    nl = sum(1 for r in rows if len([x for x in rows if x["song"] == r["song"]]) >= 8)
    print(f"is-it-{lead}'s-line, lines {nl} (all to the lead = today's ★ rule; bigger cluster = lead):")
    for k, v in lead_n.items():
        print(f"  {k:9}: {v / max(1, nl):.0%}")
    print(f"songs with 2+ singers and 8+ lines: {usable}, lines {tot}")
    print(f"  baseline (everything to the song's most frequent singer): {base / max(1, tot):.0%}")
    for k, v in ok.items():
        print(f"  2 clusters by {k:6}: {v / max(1, tot):.0%}")
    # leave-one-song-out nearest centroid (uses labels: an upper bound on how separable the voices are)
    right = n = 0
    for s in songs:
        train = [r for r in rows if r["song"] != s]
        test = [r for r in rows if r["song"] == s]
        cents = {}
        for sg in {r["singer"] for r in train}:
            cents[sg] = np.mean([r["col"] / sd for r in train if r["singer"] == sg], axis=0)
        for r in test:
            if r["singer"] not in cents:
                continue
            p = min(cents, key=lambda k: ((r["col"] / sd - cents[k]) ** 2).sum())
            right += p == r["singer"]
            n += 1
    sing = {x: sum(r["singer"] == x for r in rows) for x in sorted({r["singer"] for r in rows})}
    print(f"  leave-one-song-out centroid (uses labels, upper bound): {right / max(1, n):.0%} of {n};"
          f" most frequent singer overall {max(sing.values()) / len(rows):.0%} {sing}")


if __name__ == "__main__":
    main()
