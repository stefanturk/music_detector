#!/usr/bin/env python3
"""Section-change detector over per-bar fingerprints (features.py), scored against labels.

    python3 detect.py                    # default params, every labelled song
    python3 detect.py --tune [N]         # random search, leave-one-song-out
    python3 detect.py --show universe    # list one song's detected changes

Causal, like the rig would be: at bar j it compares the last W bars with the L bars before them,
per stem and per fingerprint (melody = chroma, rhythm = onset pattern, energy), each against its
OWN recent history (a running average of that fingerprint's distances), so no channel has to be
balanced against another. The fingerprints vote (weighted); a change fires on a local peak above
a threshold, never closer than `gap` bars to the last one, with a small bonus when it lands a
whole number of 4-bar phrases after the last change ("use the bar count when confident": only
when the grid is trusted). Two modes: "stems" (separate mixer channels) and "mix" (one line-in).
"""
import json
import math
import os
import random
import sys

import numpy as np

BASE = os.path.join(os.path.expanduser("~"), "Music", "Beat Thief Analysis")
SONGS = {  # short name -> (label name, features file)
    "universe": ("You Are The Universe - The Brand New Heavies (125.5 BPM)", "universe.json"),
    "taste": ("Taste Of You - Allen Stone (131.9 BPM)", "taste.json"),
    "clav": ("Clav it Your Way - Lettuce", "Clav it Your Way - Lettuce.json"),
    "mj": ("Don't Stop 'Til You Get Enough - Michael Jackson", "Don't Stop 'Til You Get Enough - Michael Jackson.json"),
    "jbs": ("JB's Out! - Ghost-Note (91.5 BPM)", "JB's Out! - Ghost-Note (91.5 BPM).json"),
}
EPS = 1e-6

DEFAULTS = {
    "W": 2, "L": 8, "norm": 32, "cap": 4.0, "T": 1.6, "gap": 6, "phrase_bonus": 0.3,
    "user_lag": 1.0, "tol": 4.0,
    # weights: (stem, fingerprint)
    "w_bass_melody": 1.0, "w_bass_rhythm": 0.6, "w_other_melody": 1.0, "w_other_rhythm": 0.6,
    "w_vocals_melody": 0.5, "w_vocals_energy": 0.8, "w_drums_rhythm": 0.8, "w_drums_energy": 0.8,
    "w_bass_energy": 0.5, "w_other_energy": 0.5,
    # full-mix mode
    "w_mix_melody": 1.0, "w_mix_rhythm": 0.8, "w_mix_energy": 1.0, "w_mix_bright": 0.5,
}


def cos_dist(a, b):
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def series(S, stem, fp, n):
    d = S[stem]
    if fp == "melody":
        return np.array(d["chroma"][:n])
    if fp == "rhythm":
        return np.array(d["rhythm"][:n])
    if fp == "energy":
        return np.array(d["energy"][:n])[:, None]
    return np.array(d["bright"][:n])[:, None]


def distance(x, fp, j, W, L):
    new = x[j - W + 1:j + 1].mean(axis=0)
    old = x[j - W - L + 1:j - W + 1].mean(axis=0)
    if fp in ("melody", "rhythm"):
        return cos_dist(new, old)
    return abs(math.log((new[0] + EPS) / (old[0] + EPS)))


def detect(data, p, mode="stems"):
    """Change times (seconds, start of the new part) for one song."""
    S, bars = data["stems"], data["bar_times"]
    n = min(len(v["energy"]) for v in S.values())
    if mode == "stems":
        keys = [k[2:] for k in p if k.startswith("w_") and not k.startswith("w_mix")]
    else:
        keys = [k[2:] for k in p if k.startswith("w_mix")]
    parts = []
    for k in keys:
        stem, fp = k.split("_", 1)
        if p["w_" + k] > 0:
            parts.append((p["w_" + k], series(S, stem, fp, n), fp))
    W, L, N = int(p["W"]), int(p["L"]), int(p["norm"])
    wsum = sum(w for w, _x, _f in parts) or 1.0
    ema = [None] * len(parts)
    total = np.zeros(n)
    for j in range(W + 3, n):
        if j < W + L:
            Lj = j - W                       # a short history early on
        else:
            Lj = L
        t = 0.0
        for q, (w, x, fp) in enumerate(parts):
            d = distance(x, fp, j, W, max(2, Lj))
            if ema[q] is None:
                ema[q] = d
            s = min(p["cap"], d / (ema[q] + 1e-3))
            ema[q] += (d - ema[q]) / N      # running average of this fingerprint's own distances
            t += w * s
        total[j] = t / wsum
    fires, last = [], -10 ** 6
    for j in range(W + 3, n):
        v = total[j]
        if j - last < p["gap"]:
            continue
        if last >= 0 and (j - last) % 4 == 0:
            v += p["phrase_bonus"] * 0.5
        lo, hi = max(0, j - 2), min(n, j + 3)
        if v >= p["T"] and total[j] >= total[lo:hi].max() - 1e-9:
            fires.append(j)
            last = j
    # the change began at the start of the window that just turned out different
    return [float(bars[max(0, j - W + 1)]) for j in fires]


def load_labels(name, who, lag, strong_only):
    d = json.load(open(os.path.join(BASE, "labels", "%s.%s.json" % (name, who))))
    out = [c["t"] - lag for c in d["changes"] if (not strong_only or c.get("strength", 1.0) >= 1.0)]
    return sorted(out)


def f1_against(found, truth, tol):
    used, hits = set(), 0
    for t in truth:
        best = None
        for j, s in enumerate(found):
            if j in used or abs(s - t) > tol:
                continue
            if best is None or abs(s - t) < abs(found[best] - t):
                best = j
        if best is not None:
            used.add(best)
            hits += 1
    r = hits / len(truth) if truth else 0.0
    pr = hits / len(found) if found else 0.0
    return (2 * pr * r / (pr + r) if pr + r else 0.0), pr, r


def evaluate(p, mode, songs):
    rows = {}
    for short in songs:
        name, ff = SONGS[short]
        data = json.load(open(os.path.join(BASE, "features", ff)))
        found = detect(data, p, mode)
        u = load_labels(name, "user", p["user_lag"], False)
        c = load_labels(name, "claude", 0.0, True)
        rows[short] = (found, f1_against(found, u, p["tol"]), f1_against(found, c, p["tol"]))
    return rows


def mean_f1(rows):
    return float(np.mean([(r[1][0] + r[2][0]) / 2 for r in rows.values()]))


def show(rows):
    for short, (found, fu, fc) in rows.items():
        print("  %-9s %2d found | vs you  F1 %.2f (P %.2f R %.2f) | vs me(section)  F1 %.2f (P %.2f R %.2f)"
              % (short, len(found), fu[0], fu[1], fu[2], fc[0], fc[1], fc[2]))


def tune(mode, n_iter, seed=1):
    rng = random.Random(seed)
    names = list(SONGS)
    keys = [k for k in DEFAULTS if k.startswith("w_") and (k.startswith("w_mix") == (mode == "mix"))]
    results = []
    for held in names:
        train = [s for s in names if s != held]
        best, best_p = -1.0, None
        for it in range(n_iter):
            p = dict(DEFAULTS)
            if it:
                p["T"] = rng.uniform(0.9, 3.0)
                p["W"] = rng.choice((1, 2, 3, 4))
                p["L"] = rng.choice((4, 6, 8, 12, 16))
                p["gap"] = rng.choice((4, 6, 8, 10))
                p["phrase_bonus"] = rng.choice((0.0, 0.3, 0.6, 1.0))
                p["cap"] = rng.choice((3.0, 4.0, 6.0))
                for k in keys:
                    p[k] = rng.choice((0.0, 0.3, 0.6, 1.0, 1.5))
            sc = mean_f1(evaluate(p, mode, train))
            if sc > best:
                best, best_p = sc, p
        test = evaluate(best_p, mode, [held])
        print("[%s] tuned on %s (train F1 %.2f), tested on %s:" % (mode, "+".join(train), best, held))
        show(test)
        results.append(mean_f1(test))
    print("[%s] held-out mean F1 %.2f" % (mode, float(np.mean(results))))


def main():
    a = sys.argv[1:]
    if a and a[0] == "--tune":
        n_iter = int(a[1]) if len(a) > 1 else 300
        for mode in ("stems", "mix"):
            tune(mode, n_iter)
        return
    if a and a[0] == "--show":
        data = json.load(open(os.path.join(BASE, "features", SONGS[a[1]][1])))
        for mode in ("stems", "mix"):
            print(mode, ["%.1f" % t for t in detect(data, DEFAULTS, mode)])
        return
    for mode in ("stems", "mix"):
        print("[%s] default params:" % mode)
        show(evaluate(DEFAULTS, mode, SONGS))


if __name__ == "__main__":
    main()
