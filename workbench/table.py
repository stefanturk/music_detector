#!/usr/bin/env python3
"""Print the per-bar table (energy per stem 0-9, pulse, chroma/rhythm change) for labelling by eye."""
import json, sys
import numpy as np
d = json.load(open(sys.argv[1])); bt = d["bar_times"]; S = d["stems"]
n = min(len(v["energy"]) for v in S.values())
def rel(k):
    e = np.array(S[k]["energy"][:n]); return e / (np.percentile(e, 95) + 1e-9)
def dig(x): return "%d" % min(9, int(round(x * 9)))
def change(k, field, i, w=4):
    if i < w: return 0.0
    c = np.array(S[k][field][:n]); a = c[i]; b = c[i - w:i].mean(0)
    return 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
E = {k: rel(k) for k in S}
print("bar  time   dru bas voc oth mix | pulse d/v | chromaΔ bass oth voc | rhythmΔ dru bas")
for i in range(n):
    print("%3d %5.1f   %s   %s   %s   %s   %s  |  %2d %2d    |  %.2f %.2f %.2f   |  %.2f %.2f" % (
        i + 1, bt[i], dig(E["drums"][i]), dig(E["bass"][i]), dig(E["vocals"][i]), dig(E["other"][i]),
        dig(E["mix"][i]), S["drums"]["pulse"][i], S["vocals"]["pulse"][i],
        change("bass", "chroma", i), change("other", "chroma", i), change("vocals", "chroma", i),
        change("drums", "rhythm", i), change("bass", "rhythm", i)))
