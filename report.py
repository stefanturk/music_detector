#!/usr/bin/env python3
"""Morning report for the section-change detector: one HTML page + summary.txt, built only from
the cache (features/, labels/). Nothing is played, downloaded or sent anywhere.

    python3 report.py

Proof offered, from strongest to weakest:
  1. SPLICE TEST (fully automatic, no human labels): fake DJ sets are built by gluing 16-32 bar
     chunks of different songs; the true boundaries are exact. Scored per mode at +-4 s.
  2. LABELLED SONGS: the songs the user tapped (and I labelled blind), scored as before.
  3. EVERY SONG: a chart of the detector's changes over the song's loudness, for eyeballing.
The splice test is the EASY case (different songs); changes inside one song are harder.
"""
import glob
import html
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect as D  # noqa: E402

BASE = D.BASE
TOL = 4.0


def load_all():
    out = {}
    for p in sorted(glob.glob(os.path.join(BASE, "features", "*.json"))):
        key = os.path.splitext(os.path.basename(p))[0]
        try:
            out[key] = json.load(open(p))
        except Exception:
            pass
    return out


def splice(songs, rng, n_chunks):
    """Glue chunks of different songs at the per-bar feature level. Returns (data, boundaries)."""
    keys = rng.sample(sorted(songs), min(n_chunks, len(songs)))
    stems = {k: {f: [] for f in ("energy", "chroma", "rhythm", "pulse", "bright")}
             for k in ("drums", "bass", "vocals", "other", "mix")}
    bars, bounds, t = [], [], 0.0
    for i, k in enumerate(keys):
        d = songs[k]
        n = min(len(v["energy"]) for v in d["stems"].values())
        size = rng.choice((16, 24, 32))
        if n < size + 12:
            continue
        a = rng.randrange(8, n - size)
        bt = d["bar_times"]
        if i:
            bounds.append(t)
        for j in range(a, a + size):
            dur = (bt[j + 1] - bt[j]) if j + 1 < len(bt) else (bt[j] - bt[j - 1])
            bars.append(t)
            t += dur
            for s in stems:
                for f in stems[s]:
                    stems[s][f].append(d["stems"][s][f][j])
    return {"bar_times": bars, "end": t, "stems": stems}, bounds


def splice_scores(songs, mode, sets=40, seed=7):
    rng = random.Random(seed)
    f1s, ps, rs = [], [], []
    for _ in range(sets):
        data, bounds = splice(songs, rng, rng.choice((5, 6, 7, 8)))
        if len(bounds) < 3:
            continue
        found = D.detect(data, D.DEFAULTS, mode)
        f, p, r = D.f1_against(found, bounds, TOL)
        f1s.append(f); ps.append(p); rs.append(r)
    return (float(np.mean(f1s)), float(np.mean(ps)), float(np.mean(rs)), len(f1s)) if f1s else (0, 0, 0, 0)


def chart(key, d, found_s, found_m, labels):
    W, H, M = 900, 86, 14
    end = d["end"]
    x = lambda t: M + (W - 2 * M) * t / end
    e = np.array(d["stems"]["mix"]["energy"], float)
    e = e / (e.max() + 1e-9)
    pts = " ".join("%.1f,%.1f" % (x(t), H - 18 - 52 * v) for t, v in zip(d["bar_times"], e))
    g = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="changes over time for %s">' % (W, H, html.escape(key))]
    g.append('<polyline points="%s" fill="none" stroke="var(--muted)" stroke-width="1.2"/>' % pts)
    for t in found_s:
        g.append('<line x1="%.1f" x2="%.1f" y1="4" y2="38" stroke="var(--stems)" stroke-width="2"/>' % (x(t), x(t)))
    for t in found_m:
        g.append('<line x1="%.1f" x2="%.1f" y1="40" y2="68" stroke="var(--mixc)" stroke-width="2"/>' % (x(t), x(t)))
    for who, ts in labels.items():
        col = "var(--user)" if who == "you" else "var(--me)"
        for t in ts:
            g.append('<path d="M%.1f %d l-4 8 h8z" fill="%s"/>' % (x(t), H - 12, col))
    for m in range(0, int(end) + 1, 60):
        g.append('<text x="%.1f" y="%d" font-size="9" fill="var(--muted)">%d:00</text>' % (x(m), H - 1, m // 60))
    g.append("</svg>")
    return "".join(g)


def main():
    songs = load_all()
    if not songs:
        print("no features yet"); return
    rows, charts = [], []
    lab = {s: D.SONGS[s] for s in D.SONGS}
    names = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(BASE, "features", "*.json"))}
    for key, d in songs.items():
        fs = D.detect(d, D.DEFAULTS, "stems")
        fm = D.detect(d, D.DEFAULTS, "mix")
        labels, f_you, f_me = {}, "", ""
        if key in lab:
            nm = lab[key][0]
            u = D.load_labels(nm, "user", D.DEFAULTS["user_lag"], False)
            c = D.load_labels(nm, "claude", 0.0, True)
            labels = {"you": u, "me": c}
            f_you = "%.2f / %.2f" % (D.f1_against(fs, u, TOL)[0], D.f1_against(fm, u, TOL)[0])
            f_me = "%.2f / %.2f" % (D.f1_against(fs, c, TOL)[0], D.f1_against(fm, c, TOL)[0])
        mins = d["end"] / 60.0
        rows.append((key, d["tempo"], d["grid_confidence"], mins, len(fs), len(fm), f_you, f_me))
        charts.append((key, d, fs, fm, labels))
    ss = splice_scores(songs, "stems")
    sm = splice_scores(songs, "mix")
    lines = ["Section-change detector - report (%d songs)" % len(songs),
             "SPLICE TEST (%d fake DJ sets, +-%.0f s): stems F1 %.2f (P %.2f R %.2f) | full mix F1 %.2f (P %.2f R %.2f)"
             % (ss[3], TOL, ss[0], ss[1], ss[2], sm[0], sm[1], sm[2])]
    for r in rows:
        if r[6]:
            lines.append("labelled %-10s vs you (stems/mix) %s | vs me %s" % (r[0][:10], r[6], r[7]))
    per_min_s = np.mean([r[4] / r[3] for r in rows]); per_min_m = np.mean([r[5] / r[3] for r in rows])
    lines.append("changes per minute, average over all songs: stems %.2f, full mix %.2f" % (per_min_s, per_min_m))
    os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)
    open(os.path.join(BASE, "reports", "summary.txt"), "w").write("\n".join(lines) + "\n")

    h = ["""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Change detector report</title><style>
:root{--bg:#fff;--fg:#1b1b1f;--muted:#8a8a93;--card:#f4f4f7;--stems:#e07b00;--mixc:#2166d1;--user:#1a9850;--me:#8e44ad}
@media (prefers-color-scheme:dark){:root{--bg:#141418;--fg:#ececf1;--muted:#8a8a96;--card:#1e1e25;--stems:#ffa63d;--mixc:#6aa3ff;--user:#4cd07d;--me:#c58bf0}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}
h1{font-size:22px}.card{background:var(--card);border-radius:10px;padding:12px 14px;margin:12px 0}
.big{font-size:28px;font-weight:650}.k{display:inline-block;margin-right:14px;font-size:13px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:4px 6px;text-align:left;border-bottom:1px solid var(--muted)}
.n{font-weight:600;font-size:13px;margin:0 0 2px}</style>"""]
    h.append("<h1>Section-change detector: how well does it work?</h1>")
    h.append('<div class="card"><b>Test 1: fake DJ sets with known change times</b> (no human labels; the easy case)<br>'
             '<span class="big">stems %.0f%%</span> &nbsp; <span class="big">full mix %.0f%%</span>'
             '<br><span class="k">score = F1 within &plusmn;%.0f s, %d sets, %d songs</span>'
             '<span class="k">stems: found %.0f%% of real changes, %.0f%% of its alarms were real</span>'
             '<span class="k">mix: %.0f%% / %.0f%%</span></div>'
             % (ss[0] * 100, sm[0] * 100, TOL, ss[3], len(songs), ss[2] * 100, ss[1] * 100, sm[2] * 100, sm[1] * 100))
    lab_rows = [r for r in rows if r[6]]
    if lab_rows:
        h.append('<div class="card"><b>Test 2: songs you tapped / I labelled blind</b> (F1 stems / mix)<table><tr><th>song<th>vs you<th>vs me</tr>')
        for r in lab_rows:
            h.append("<tr><td>%s<td>%s<td>%s</tr>" % (html.escape(r[0]), r[6], r[7]))
        h.append("</table></div>")
    h.append('<div class="card"><span class="k"><b style="color:var(--stems)">&#9475;</b> stems detector</span>'
             '<span class="k"><b style="color:var(--mixc)">&#9475;</b> full-mix detector</span>'
             '<span class="k"><b style="color:var(--user)">&#9650;</b> your taps (lag-corrected)</span>'
             '<span class="k"><b style="color:var(--me)">&#9650;</b> my labels</span>'
             '<span class="k">grey line = loudness</span></div>')
    h.append("<h2>Every song (%d)</h2>" % len(rows))
    for (key, d, fs, fm, labels), r in zip(charts, rows):
        h.append('<div class="card"><p class="n">%s</p><span class="k">%.0f BPM &middot; grid confidence %.1f &middot; %.1f min &middot; changes: stems %d, mix %d</span>%s</div>'
                 % (html.escape(key), r[1], r[2], r[3], r[4], r[5], chart(key, d, fs, fm, labels)))
    h.append('<p class="k">Not in this report yet: energy row (Dark/Standard/Crazy) and colour mood. Settings are the untuned defaults.</p>')
    open(os.path.join(BASE, "reports", "index.html"), "w").write("\n".join(h))
    print("\n".join(lines))
    print("report: %s" % os.path.join(BASE, "reports", "index.html"))


if __name__ == "__main__":
    main()
