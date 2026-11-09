#!/usr/bin/env python3
"""Live-band change detector over per-channel levels (multitrack.py features).

    python3 live.py <dir>        # events list + chart with the events drawn on it

Causal (each decision only uses the past), so it is what the rig could run on the mixer's
channels. Per channel: "now" = mean level of the last 2 s; "before" = median level from 12 s
to 4 s ago. A channel ENTERS when now is >= ENTER_DB above before (and above its own
silence guard), and EXITS when it is that far below. Entries and exits within a few seconds of
each other are one EVENT. Then a rule turns an event into a change strength by who moved:
horns together, a singer, the rhythm section (drums + bass), a lead.
"""
import json
import os
import sys

import numpy as np

SMOOTH = 6          # frames (3 s) median, so staccato hits do not look like "off"
HYST_DB = 2.0
DEBOUNCE = 14       # frames (7 s): a new arrangement must hold this long to count
MIN_GAP = 12        # frames (6 s) between events
WEIGHT = {"drums": 1.0, "bass": 1.0, "guitar": 0.7, "vox": 1.0, "horns": 1.2}


def channel_on(db):
    """Is this channel playing? Threshold halfway (in dB) between its own quiet and loud
    levels, with hysteresis, on a 3 s median."""
    db = np.asarray(db, float)
    sm = np.array([np.median(db[max(0, i - SMOOTH + 1):i + 1]) for i in range(len(db))])
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    thr = lo + 0.5 * (hi - lo)
    on, state = np.zeros(len(db), bool), False
    for i, v in enumerate(sm):
        if state and v < thr - HYST_DB:
            state = False
        elif not state and v > thr + HYST_DB:
            state = True
        on[i] = state
    return on


def arrangement(data):
    """Per frame: how many channels of each group are playing."""
    ch = data["channels"]
    groups = sorted({c["group"] for c in ch.values()})
    n = len(next(iter(ch.values()))["db"])
    cnt = {g: np.zeros(n, int) for g in groups}
    on_by = {}
    for name, c in ch.items():
        on = channel_on(c["db"])
        on_by[name] = on
        cnt[c["group"]] += on
    return groups, cnt, on_by


def events(data):
    """Arrangement changes while a song is playing, plus song start/end.

    Song = drums AND bass both playing. Between songs (talking, tuning) nothing but the song
    start/end itself counts. A new arrangement must hold DEBOUNCE frames before it fires, so a
    horn line that drops out for two bars and comes back is not two changes."""
    groups, cnt, _on = arrangement(data)
    n = len(cnt[groups[0]])
    hop = data["hop_s"]

    def state(i):
        return {g: (int(cnt[g][i]) if g in ("vox", "horns", "guitar") else int(cnt[g][i] > 0)) for g in groups}

    def in_song(i):
        return cnt["drums"][i] > 0 and cnt["bass"][i] > 0

    out, last = [], -10 ** 6
    stable, song = state(0), in_song(0)
    i = 1
    while i < n - DEBOUNCE:
        hold = [in_song(j) for j in range(i, i + DEBOUNCE)]
        if in_song(i) != song and all(h == in_song(i) for h in hold):
            song = in_song(i)
            stable = state(i)
            out.append({"t": i * hop, "score": 6.0, "what": "SONG START" if song else "SONG END"})
            last = i
        elif song and i - last >= MIN_GAP:
            cand = state(i)
            if cand != stable and all(state(j) == cand for j in range(i, i + DEBOUNCE)):
                moved = [(g, cand[g] - stable[g]) for g in groups if cand[g] != stable[g]]
                score = sum(WEIGHT.get(g, 0.5) * abs(dv) for g, dv in moved)
                what = ", ".join("%s %s%d" % (g, "+" if dv > 0 else "", dv) for g, dv in moved)
                out.append({"t": i * hop, "score": round(score, 2), "what": what})
                stable, last = cand, i
        i += 1
    return out



# ---- section changes: each pitched channel's chroma, last 4 s vs the 16 s before it -----------
W_S, L_S, EMA_N, SECTION_T, SECTION_GAP_S, LEVEL_W = 6, 20, 120, 1.1, 6, 0.5   # frames of 0.5 s


def _cosd(a, b):
    return 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def section_novelty(data):
    """Every channel votes with how much its own sound changed, each against its OWN running
    average, so no channel has to be balanced against another: chroma (pitch content) for the
    pitched channels, plus level (LEVEL_W as much) for all of them."""
    n = len(next(iter(data["channels"].values()))["db"])
    tot, wsum = np.zeros(n), 0.0
    for c in data["channels"].values():
        series = [("chroma", np.array(c["chroma"]), 1.0)] if "chroma" in c else []
        series.append(("level", np.array(c["db"])[:, None], LEVEL_W))
        for kind, x, w in series:
            ema = None
            for j in range(W_S + L_S, n):
                new, old = x[j - W_S + 1:j + 1].mean(0), x[j - W_S - L_S + 1:j - W_S + 1].mean(0)
                d = _cosd(new, old) if kind == "chroma" else abs(float(new[0] - old[0]))
                ema = d if ema is None else ema
                tot[j] += w * min(4.0, d / (ema + 1e-3))
                ema += (d - ema) / EMA_N
            wsum += w
    return tot / wsum


def section_changes(data, T=SECTION_T):
    tot = section_novelty(data)
    gap = int(SECTION_GAP_S / data["hop_s"])
    out, last = [], -10 ** 6
    for j in range(2, len(tot) - 3):
        if j - last >= gap and tot[j] >= T and tot[j] >= tot[max(0, j - 4):j + 5].max() - 1e-9:
            out.append(round((j - W_S // 2) * data["hop_s"], 1))
            last = j
    return out


def mix_visible(data, evs, win_s=3.0):
    """How many events show up as a clear step in the SUMMED level (a one-line mix)?"""
    ch = data["channels"]
    p = sum(10 ** (np.array(c["db"]) / 10) for c in ch.values())
    mix = 10 * np.log10(p + 1e-12)
    hop = data["hop_s"]
    vis = []
    for e in evs:
        i = int(e["t"] / hop)
        a = mix[max(0, i - 20):max(1, i - 2)]
        b = mix[i + 2:i + 2 + int(win_s / hop)]
        vis.append(abs(float(np.median(b) - np.median(a))) if len(a) and len(b) else 0.0)
    return vis


def draw(d, data, evs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(data["channels"])
    n = len(data["channels"][names[0]]["db"])
    t = np.arange(n) * data["hop_s"] / 60
    _g, _c, on_by = arrangement(data)
    fig, ax = plt.subplots(len(names), 1, figsize=(18, 0.55 * len(names) + 1), sharex=True)
    for a, nm in zip(ax, names):
        db = np.array(data["channels"][nm]["db"])
        hi = np.percentile(db, 99)
        a.fill_between(t, np.clip(db, hi - 50, hi), hi - 50, lw=0, color="#999")
        a.fill_between(t, hi - 50, hi - 50 + 50 * on_by[nm], lw=0, color="#2a7", alpha=0.45)
        a.set_ylabel(nm[3:12], rotation=0, ha="right", va="center", fontsize=7)
        a.set_yticks([])
        a.margins(x=0)
        for e in evs:
            a.axvline(e["t"] / 60, color="#e8a" if e["score"] < 1.5 else "#e50", lw=0.8 if e["score"] < 1.5 else 1.4)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "live_events.png"), dpi=70)


if __name__ == "__main__":
    d = os.path.expanduser(sys.argv[1])
    data = json.load(open(os.path.join(d, "mt_features.json")))
    evs = events(data)
    vis = mix_visible(data, evs)
    for e, v in zip(evs, vis):
        print("%5.1f min  score %.1f  mix-step %4.1f dB  %s" % (e["t"] / 60, e["score"], v, e["what"]))
    big = [(e, v) for e, v in zip(evs, vis) if e["score"] >= 1.5]
    print("\n%d events, %d strong (score>=1.5); of the strong ones %d show >=3 dB in the summed mix"
          % (len(evs), len(big), sum(1 for _e, v in big if v >= 3)))
    json.dump(evs, open(os.path.join(d, "live_events.json"), "w"), indent=1)
    sec = section_changes(data)
    json.dump(sec, open(os.path.join(d, "live_sections.json"), "w"))
    print("section changes (chroma novelty): %d, at %s" % (len(sec), [round(t) for t in sec]))
    draw(d, data, evs)
