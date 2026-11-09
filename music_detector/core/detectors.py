"""What each instrument is doing, frame by frame. Deterministic, no AI, numpy only.

Every detector reads per-voice ACTIVITY: a voice's loudness on 50 ms frames divided by its own
featured level (see whoplays), so 1.0 means "playing at about its usual full level". The same
framework runs for every voice; only small parts are tailored (drum fills read the tom mics, the
vocal is judged on lyric lines when there are any).

Studio mode runs these over a whole song. The per-frame parts are written as small state machines
(`Hysteresis.step`) so a live stream can later feed the same steps one frame at a time.

Times in and out are in frames (HOP seconds) or seconds where the name says `_s`.
"""
import numpy as np

HOP = 0.05                # frame length, seconds (the whoplays envelope hop)
BIN_S = 0.5               # the energy / featured curves are reported on bins this long

# playing / not playing
PLAY_ON = 0.30            # a voice turns ON at this activity...
PLAY_OFF = 0.18           # ...and OFF only under this...
PLAY_ON_S = 0.2           # ...held this long (on)
PLAY_OFF_S = 1.0          # ...or this long (off), so a breath or a staccato rest isn't "stopped"

# energy curve
ENERGY_SMOOTH_S = 4.0     # moving-average window
ENERGY_PCT = (10, 90)     # scaled so the song's 10th percentile is 0 and its 90th is 1

# featured score
FEAT_SHARE_W = 0.6        # weight of "a bigger share of the band than usual"
FEAT_LEVEL_W = 0.4        # weight of "louder than its own usual level"
FEAT_CAP = 1.0            # each part saturates at this much above usual (x2 share / x2 level)

# drum fills
FILL_TOM_ON = 0.30        # tom mics at this fraction of their loudest moments in the song...
FILL_TOM_VS_KS = 0.6      # ...and at least this fraction of kick+snare's level (toms take over)
FILL_GAP_S = 0.35         # tom hits closer than this belong to one fill
FILL_MIN_S = 0.4          # a fill lasts at least this long...
FILL_MAX_S = 6.0          # ...and at most this long (longer is a tom groove, not a fill)
FILL_BIG = 0.6            # strength from here up is a "big" fill
BURST_WIN_S = 1.0         # no tom mics: drum onsets counted per this window...
BURST_DENSITY = 2.0       # ...a burst is this many times the song's usual onset rate...
BURST_LEVEL = 1.3         # ...while the drums are this many times their usual level

# solos
SOLO_MIN_S = 4.0          # a solo lasts at least this long
SOLO_DOM = 1.3            # the soloist is at least this many times louder than the next voice
SOLO_GAP_S = 0.75         # short dips inside a solo are bridged
RHYTHM = ("DRUMS", "BASS")

# entrances
ENTRANCE_OFF_S = 3.0      # a voice "enters" when it starts after at least this long off

# best moments
BEST_WIN_S = 4.0          # a best moment is the best-scoring stretch this long
BEST_MIN = 0.25           # ...whose mean featured score is at least this
BEST_N = 3                # up to this many per voice


# ---------- small state machines (the live-ready part) ----------

class Hysteresis:
    """On/off with two thresholds and hold times. `step(x)` takes one frame, returns the state.
    `onset` is the frame index where the current ON run started (backfilled to its first frame)."""

    def __init__(self, on=PLAY_ON, off=PLAY_OFF, on_s=PLAY_ON_S, off_s=PLAY_OFF_S, hop=HOP):
        self.on, self.off = on, off
        self.on_n = max(1, int(round(on_s / hop)))
        self.off_n = max(1, int(round(off_s / hop)))
        self.state, self.above, self.below, self.i = False, 0, 0, -1
        self.onset = None

    def step(self, x):
        self.i += 1
        if not self.state:
            self.above = self.above + 1 if x >= self.on else 0
            if self.above >= self.on_n:
                self.state, self.below = True, 0
                self.onset = self.i - self.above + 1
        else:
            self.below = self.below + 1 if x < self.off else 0
            if self.below >= self.off_n:
                self.state, self.above = False, 0
        return self.state


def playing_mask(act, **kw):
    """Frame-by-frame playing state for one voice; the trailing off-hold is trimmed so a run ends
    where the voice actually went quiet."""
    h = Hysteresis(**kw)
    out = np.zeros(len(act), dtype=bool)
    run_start = None
    for i, x in enumerate(act):
        on = h.step(float(x))
        if on and run_start is None:
            run_start = h.onset
            out[run_start:i + 1] = True
        elif on:
            out[i] = True
        elif run_start is not None:                  # just turned off: drop the quiet hold
            out[max(run_start, i - h.off_n + 1):i + 1] = False
            run_start = None
    return out


# ---------- helpers ----------

def runs(mask, gap=0, min_len=1):
    """[(start, end)] frame runs of True, bridging gaps of <= `gap` frames, keeping runs >= min_len."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if out and i - out[-1][1] <= gap:
            out[-1] = (out[-1][0], j)
        else:
            out.append((i, j))
        i = j
    return [(a, b) for a, b in out if b - a >= min_len]


def _bins(x, hop=HOP, bin_s=BIN_S):
    k = max(1, int(round(bin_s / hop)))
    n = len(x) // k
    if n == 0:
        return np.array([float(np.mean(x))]) if len(x) else np.zeros(0)
    return x[:n * k].reshape(n, k).mean(axis=1)


def _smooth(x, n):
    if n <= 1 or len(x) == 0:
        return np.asarray(x, dtype=float)
    k = np.ones(n) / n
    pad = np.pad(np.asarray(x, dtype=float), (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def _sec(f, hop=HOP):
    return round(f * hop, 2)


# ---------- detectors ----------

def energy_curve(act, hop=HOP, bin_s=BIN_S):
    """Whole-band intensity per bin, 0..1 over the song: summed activity plus how busy it is
    (positive changes in the summed level, i.e. onsets), smoothed over ENERGY_SMOOTH_S."""
    voices = list(act)
    if not voices:
        return np.zeros(0)
    tot = np.sum([np.asarray(act[v], dtype=float) for v in voices], axis=0)
    flux = np.maximum(np.diff(tot, prepend=tot[:1]), 0.0)
    level = _bins(tot, hop, bin_s)
    busy = _bins(flux, hop, bin_s)
    if busy.max() > 0:
        busy = busy / (np.percentile(busy, 90) + 1e-9) * (np.percentile(level, 90) + 1e-9)
    e = _smooth(level + 0.5 * busy, max(1, int(round(ENERGY_SMOOTH_S / bin_s))))
    lo, hi = np.percentile(e, ENERGY_PCT[0]), np.percentile(e, ENERGY_PCT[1])
    return np.clip((e - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def featured_scores(act, playing, hop=HOP, bin_s=BIN_S):
    """{voice: per-bin score 0..1}: how much this voice stands out right now, judged against itself.
    Two parts: its SHARE of the band versus its usual share in this song (steady keys score when the
    guitars and horns drop out and they fill the space), and its level versus its own usual level.
    Zero while it isn't playing."""
    voices = list(act)
    a = {v: _bins(np.asarray(act[v], dtype=float), hop, bin_s) for v in voices}
    p = {v: _bins(playing[v].astype(float), hop, bin_s) >= 0.5 for v in voices}
    tot = np.sum([a[v] for v in voices], axis=0) + 1e-9
    out = {}
    for v in voices:
        share = a[v] / tot
        on = p[v]
        if not np.any(on):
            out[v] = np.zeros(len(share))
            continue
        usual_share = float(np.median(share[on])) + 1e-9
        usual_level = float(np.median(a[v][on])) + 1e-9
        s_part = np.clip(share / usual_share - 1.0, 0.0, FEAT_CAP) / FEAT_CAP
        l_part = np.clip(a[v] / usual_level - 1.0, 0.0, FEAT_CAP) / FEAT_CAP
        out[v] = np.where(on, FEAT_SHARE_W * s_part + FEAT_LEVEL_W * l_part, 0.0)
    return out


def _norm_top(x, pct=99.5):
    x = np.asarray(x, dtype=float)
    return x / (np.percentile(x, pct) + 1e-9) if len(x) else x


def drum_fills(toms=None, kick_snare=None, drums=None, hop=HOP):
    """Drum fills: [{start, end, strength, toms, confidence}] in seconds.

    With tom mics (`toms` = summed tom envelope, `kick_snare` = summed kick + snare): a fill is where
    the toms come up to FILL_TOM_ON of their loudest in the song AND match kick+snare's level, i.e.
    the drummer moved onto the toms. Without them (`drums` = the whole kit's envelope): a burst of
    drum onsets well above the song's usual rate, at above-usual level, flagged low confidence."""
    out = []
    if toms is not None and len(toms) and np.max(toms) > 0:
        t = _norm_top(toms)
        ks = _norm_top(kick_snare) if kick_snare is not None and len(kick_snare) else np.zeros(len(t))
        n = min(len(t), len(ks))
        t, ks = t[:n], ks[:n]
        mask = (t >= FILL_TOM_ON) & (t >= FILL_TOM_VS_KS * ks)
        for a, b in runs(mask, gap=int(round(FILL_GAP_S / hop)), min_len=1):
            dur = (b - a) * hop
            if dur < FILL_MIN_S or dur > FILL_MAX_S:
                continue
            seg_t = float(np.mean(t[a:b]))
            ratio = float(np.mean(t[a:b] / (ks[a:b] + t[a:b] + 1e-9)))
            out.append({"start": _sec(a, hop), "end": _sec(b, hop),
                        "strength": round(min(1.0, seg_t * (0.5 + ratio)), 2),
                        "toms": round(ratio, 2), "confidence": "high"})
        return out
    if drums is None or not len(drums):
        return out
    d = np.asarray(drums, dtype=float)
    flux = np.maximum(np.diff(d, prepend=d[:1]), 0.0)
    thr = np.percentile(flux, 90)
    peaks = (flux > thr) & (flux >= np.roll(flux, 1)) & (flux >= np.roll(flux, -1))
    w = max(1, int(round(BURST_WIN_S / hop)))
    dens = np.convolve(peaks.astype(float), np.ones(w), mode="same")
    lvl = np.convolve(d, np.ones(w) / w, mode="same")
    usual_d, usual_l = np.median(dens) + 1e-9, np.median(lvl) + 1e-9
    mask = (dens >= BURST_DENSITY * usual_d) & (lvl >= BURST_LEVEL * usual_l)
    for a, b in runs(mask, gap=int(round(FILL_GAP_S / hop)), min_len=1):
        dur = (b - a) * hop
        if dur < FILL_MIN_S or dur > 4.0:
            continue
        out.append({"start": _sec(a, hop), "end": _sec(b, hop),
                    "strength": round(min(1.0, float(np.mean(lvl[a:b])) / (3 * usual_l)), 2),
                    "toms": None, "confidence": "low"})
    return out


def singing_mask(n, lines=None, vocal_act=None, hop=HOP):
    """Frames where someone sings: the lyric lines when given (song seconds), else the VOCAL voice's
    playing state."""
    m = np.zeros(n, dtype=bool)
    if lines:
        for s, e in lines:
            m[max(0, int(s / hop)):max(0, int(np.ceil(e / hop)))] = True
        return m
    if vocal_act is not None:
        return playing_mask(vocal_act)[:n]
    return m


def solos(act, playing, singing, player_of=None, hop=HOP):
    """[{voice, player, start, end, confidence}]: one voice clearly out front for >= SOLO_MIN_S while
    nobody sings. A melodic voice must beat every other melodic voice by SOLO_DOM; the drums or bass
    need everyone else (melodic) under PLAY_OFF."""
    voices = [v for v in act if v != "VOCAL"]
    n = len(singing)
    out = []
    for v in voices:
        me = np.asarray(act[v], dtype=float)[:n]
        others = [np.asarray(act[o], dtype=float)[:n] for o in act
                  if o != v and o != "VOCAL" and (o not in RHYTHM or v in RHYTHM)]
        if v in RHYTHM:
            mel = [np.asarray(act[o], dtype=float)[:n] for o in act if o not in RHYTHM and o != v]
            quiet = np.max(mel, axis=0) < PLAY_OFF if mel else np.ones(n, dtype=bool)
            mask = playing[v][:n] & quiet & ~singing
            dom = me / (np.max(mel, axis=0) + 0.05) if mel else me
        else:
            top = np.max(others, axis=0) if others else np.zeros(n)
            mask = playing[v][:n] & (me >= SOLO_DOM * top) & ~singing
            dom = me / (top + 0.05)
        for a, b in runs(mask, gap=int(round(SOLO_GAP_S / hop)),
                         min_len=int(round(SOLO_MIN_S / hop))):
            conf = float(np.clip((np.median(dom[a:b]) - 1.0) / 2.0, 0.0, 1.0))
            out.append({"voice": v, "player": (player_of or {}).get(v), "start": _sec(a, hop),
                        "end": _sec(b, hop), "confidence": round(conf, 2)})
    return sorted(out, key=lambda s: (s["start"], s["voice"]))


def entrances(act, playing, hop=HOP):
    """[{voice, at}]: a voice starting after >= ENTRANCE_OFF_S off. `at` is where its level starts
    rising (the first frame of the rise into the ON run), in seconds."""
    off_n = int(round(ENTRANCE_OFF_S / hop))
    out = []
    for v, m in playing.items():
        x = np.asarray(act[v], dtype=float)
        rs = runs(m)
        prev_end = None
        for a, b in rs:
            if (prev_end is None and a >= off_n) or (prev_end is not None and a - prev_end >= off_n):
                i = a
                peak = float(np.max(x[a:min(b, a + int(round(0.5 / hop)) + 1)]))
                # back up through the rise: frames still above a fifth of where it gets to
                while i > 0 and x[i - 1] >= 0.2 * peak and x[i - 1] < x[i] + 1e-12:
                    i -= 1
                out.append({"voice": v, "at": _sec(i, hop)})
            prev_end = b
    return sorted(out, key=lambda e: (e["at"], e["voice"]))


def line_singers(lines, mic_act, mic_player=None, hop=HOP):
    """For each lyric line (song seconds), the vocal mic that is loudest over it (each mic judged on
    its own scale) -> [{start, end, stem, player}]. `mic_act` = {stem: activity}."""
    out = []
    for s, e in lines or []:
        a, b = max(0, int(s / hop)), max(1, int(np.ceil(e / hop)))
        best, best_v = None, -1.0
        for stem in sorted(mic_act):
            x = np.asarray(mic_act[stem], dtype=float)[a:b]
            val = float(np.mean(x)) if len(x) else 0.0
            if val > best_v:
                best, best_v = stem, val
        out.append({"start": round(float(s), 2), "end": round(float(e), 2), "stem": best,
                    "player": (mic_player or {}).get(best)})
    return out


def best_moments(featured, bin_s=BIN_S, player_of=None):
    """{voice: [{start, end, score, player}]}: each voice's best stretches of BEST_WIN_S (highest mean
    featured score, >= BEST_MIN), up to BEST_N that don't overlap, best first."""
    w = max(1, int(round(BEST_WIN_S / bin_s)))
    out = {}
    for v, f in featured.items():
        f = np.asarray(f, dtype=float)
        if len(f) < w:
            out[v] = []
            continue
        mean = np.convolve(f, np.ones(w) / w, mode="valid")
        taken, picks = np.zeros(len(f), dtype=bool), []
        for i in np.argsort(-mean, kind="stable"):
            if mean[i] < BEST_MIN or len(picks) >= BEST_N:
                break
            if taken[i:i + w].any():
                continue
            taken[i:i + w] = True
            picks.append({"start": round(i * bin_s, 2), "end": round((i + w) * bin_s, 2),
                          "score": round(float(mean[i]), 2), "player": (player_of or {}).get(v)})
        out[v] = picks
    return out


def trading(spans):
    """[{pair, start, end}] from who's-playing spans tagged as a trading section (whoplays' rule)."""
    out = []
    for sp in spans or []:
        pair = sp.get("trading")
        if not pair:
            continue
        if out and out[-1]["pair"] == pair and abs(out[-1]["end"] - sp["start_s"]) < 0.11:
            out[-1]["end"] = sp["end_s"]
        else:
            out.append({"pair": pair, "start": sp["start_s"], "end": sp["end_s"]})
    return out
