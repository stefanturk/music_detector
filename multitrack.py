#!/usr/bin/env python3
"""Per-channel features for a multitrack live recording (one mono wav per mixer channel).

    python3 multitrack.py features <dir>      # writes <dir>/mt_features.json
    python3 multitrack.py chart <dir>         # writes <dir>/mt_chart.png

Per channel, every 0.5 s: level (dB, RMS) and, for pitched channels, a 12-bin chroma.
Channels are grouped by the name of the file: drums (kick/snare/oh/tom), bass, guitar,
vox, horns (trumpet/sax). Nothing here knows which song is playing.
"""
import json
import os
import sys

import librosa
import numpy as np

SR = 22050
HOP = SR // 2           # 0.5 s frames
GROUPS = [("drums", ("KICK", "SNARE", "OH ", "TOM")), ("bass", ("BASS",)),
          ("guitar", ("GUITAR",)), ("vox", ("VOX",)), ("horns", ("TRUMPET", "SAX"))]
PITCHED = ("bass", "guitar", "vox", "horns")


def group_of(name):
    for g, keys in GROUPS:
        if any(k in name.upper() for k in keys):
            return g
    return "other"


def features(d):
    out = {"hop_s": HOP / SR, "channels": {}}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".wav") or f.startswith("mt_"):
            continue
        name = os.path.splitext(f)[0]
        y, _ = librosa.load(os.path.join(d, f), sr=SR, mono=True)
        n = len(y) // HOP
        rms = np.sqrt(np.mean(y[:n * HOP].reshape(n, HOP) ** 2, axis=1))
        g = group_of(name)
        ch = {"group": g, "db": (20 * np.log10(rms + 1e-7)).round(2).tolist()}
        if g in PITCHED:
            c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP,
                                           fmin=librosa.note_to_hz("C2" if g == "bass" else "C3"),
                                           n_octaves=3 if g == "bass" else 5)
            c = c[:, :n].T
            ch["chroma"] = (c / (c.sum(axis=1, keepdims=True) + 1e-9)).round(3).tolist()
        out["channels"][name] = ch
        print("%-24s %-7s p95 %.1f dB" % (name, g, np.percentile(ch["db"], 95)))
    return out


def chart(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.load(open(os.path.join(d, "mt_features.json")))
    names = list(data["channels"])
    n = len(data["channels"][names[0]]["db"])
    t = np.arange(n) * data["hop_s"] / 60
    fig, ax = plt.subplots(len(names), 1, figsize=(18, 0.55 * len(names) + 1), sharex=True)
    for a, nm in zip(ax, names):
        db = np.array(data["channels"][nm]["db"])
        hi = np.percentile(db, 99)
        a.fill_between(t, np.clip(db, hi - 50, hi), hi - 50, lw=0, color="#444")
        a.set_ylabel(nm[3:12], rotation=0, ha="right", va="center", fontsize=7)
        a.set_yticks([])
        a.margins(x=0)
    ax[-1].set_xlabel("minutes into the 15-minute cut")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "mt_chart.png"), dpi=70)


if __name__ == "__main__":
    cmd, d = sys.argv[1], os.path.expanduser(sys.argv[2])
    if cmd == "features":
        json.dump(features(d), open(os.path.join(d, "mt_features.json"), "w"))
    elif cmd == "chart":
        chart(d)
