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
# ...and judged on each player's OWN usual (with the part's sound, timbre.measure):
SOLO_SMOOTH_S = 4.0       # level / featured / sound are averaged over this long
SOLO_OWN_UP = 1.2         # up on the player's own usual level (median while playing in the song)
SOLO_FEAT_MIN = 0.35      # ...or the most featured part: at least this featured...
SOLO_FEAT_LEAD = 1.4      # ...and this many times the next part's featured score
SOLO_CLAR_UP = 0.15       # a lead line: a clearer single pitch than the part's own usual by this...
SOLO_VIB = 0.15           # ...or this share of held notes bent / with vibrato
SOLO_STRONG_CLAR = 0.25   # a strong lead sound (this much clearer AND bends) counts at usual level...
SOLO_SOUND_ONLY_S = 8.0   # ...when it lasts this long
SOLO_LEAD_GAP_S = 2.5     # another player's answering phrase inside a solo doesn't end it
SOLO_LEAD_MIN_S = 6.0     # an own-usual solo lasts at least this long

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


def _lead_solos(act, playing, singing, lead, featured, hop=HOP, bin_s=BIN_S):
    """{voice: [(start_frame, end_frame, confidence)]}: melodic solos judged against each player's
    OWN usual, not against the other players' loudness (Stefan, Sure Dont Miss Me: his guitar solo
    broke up because Chris's keys comping was as loud). A half-second bin is a solo bin when nobody
    sings, the part sounds like a lead line (a clearer single pitch than its own usual, or bends /
    vibrato) AND it is up on its own usual level or the clearly most featured part. A long stretch
    of a strong lead sound counts at its usual level. Where two parts claim a bin, the more featured
    one keeps it."""
    k = max(1, int(round(bin_s / hop)))
    nb = min([len(f) for f in featured.values()] + [len(singing) // k])
    sm = lambda x: _smooth(np.asarray(x, dtype=float)[:nb], max(1, int(round(SOLO_SMOOTH_S / bin_s))))
    sing_b = _bins(np.asarray(singing, dtype=float), hop, bin_s)[:nb] > 0.3
    fs = {v: sm(f) for v, f in featured.items()}
    claim, conf_b = {}, {}
    for v, t in (lead or {}).items():
        if v in RHYTHM or v == "VOCAL" or v not in act or v not in fs:
            continue
        lvl = _bins(np.asarray(act[v], dtype=float), hop, bin_s)[:nb]
        pl = _bins(playing[v].astype(float), hop, bin_s)[:nb] > 0.5
        clar = np.asarray(t.get("clarity", []), dtype=float)[:nb]
        vib = np.asarray(t.get("vibrato", []), dtype=float)[:nb]
        tp = np.asarray(t.get("playing", []), dtype=float)[:nb] > 0.2
        if pl.sum() < 10 or len(clar) < nb or len(vib) < nb or not tp.any():
            continue
        own = sm(lvl) / (np.median(lvl[pl]) + 1e-9)
        others = [fs[o] for o in fs if o not in (v, "VOCAL")]
        top = np.max(others, axis=0) if others else np.zeros(nb)
        flead = (fs[v] >= SOLO_FEAT_MIN) & (fs[v] >= SOLO_FEAT_LEAD * top)
        margin = sm(clar) - float(np.median(clar[tp]))
        vs = sm(vib)
        sound = (margin >= SOLO_CLAR_UP) | (vs >= SOLO_VIB)
        base = pl & ~sing_b
        m = base & sound & ((own >= SOLO_OWN_UP) | flead)
        strong = base & (margin >= SOLO_STRONG_CLAR) & (vs >= SOLO_VIB)
        for a, b in runs(strong, gap=int(round(SOLO_LEAD_GAP_S / bin_s)),
                         min_len=int(round(SOLO_SOUND_ONLY_S / bin_s))):
            m[a:b] = True
        claim[v] = m
        conf_b[v] = np.clip(0.3 + 0.35 * np.clip((own - 1.0) / 0.5, 0, 1)
                            + 0.35 * np.clip(np.maximum(margin / 0.3, vs / 0.4), 0, 1), 0, 1)
    if len(claim) > 1:                       # one soloist per bin: the most featured part keeps it
        vs_ = sorted(claim)
        stack = np.array([np.where(claim[v], fs[v], -1.0) for v in vs_])
        win = np.argmax(stack, axis=0)
        for i, v in enumerate(vs_):
            claim[v] = claim[v] & (win == i)
    out = {}
    for v, m in claim.items():
        for a, b in runs(m, gap=int(round(SOLO_LEAD_GAP_S / bin_s)),
                         min_len=int(round(SOLO_LEAD_MIN_S / bin_s))):
            out.setdefault(v, []).append((a * k, b * k, round(float(np.mean(conf_b[v][a:b])), 2)))
    return out


def solos(act, playing, singing, player_of=None, hop=HOP, lead=None, featured=None):
    """[{voice, player, start, end, confidence}]: one voice clearly out front for >= SOLO_MIN_S while
    nobody sings. A melodic voice must beat every other melodic voice by SOLO_DOM; the drums or bass
    need everyone else (melodic) under PLAY_OFF.

    `lead` = {voice: timbre.measure(...) of its loudest stem} (per BIN_S bin) adds the own-usual
    rule (_lead_solos): those solos join and extend the loudness ones, and a loudness-only solo
    lying mostly inside another part's lead solo is dropped."""
    voices = [v for v in act if v != "VOCAL"]
    n = len(singing)
    if lead:
        if featured is None:
            featured = featured_scores(act, playing, hop)
        ls = _lead_solos(act, playing, singing, lead, featured, hop)
        old = solos(act, playing, singing, None, hop)
        merged = []
        for v in voices:
            mask = np.zeros(n, dtype=bool)
            conf = np.zeros(n)
            for a, b, c in ls.get(v, []):
                mask[a:b] = True
                conf[a:b] = np.maximum(conf[a:b], c)
            for s in old:
                if s["voice"] != v:
                    continue
                a, b = int(round(s["start"] / hop)), int(round(s["end"] / hop))
                inside = sum(max(0, min(b, y) - max(a, x)) for o, segs in ls.items() if o != v
                             for x, y, _ in segs)
                if inside > 0.5 * (b - a):
                    continue                 # another player's solo, really
                mask[a:b] = True
                conf[a:b] = np.maximum(conf[a:b], s["confidence"])
            for a, b in runs(mask):
                merged.append((a, b, v, round(float(np.max(conf[a:b])), 2)))
        # one soloist at a time: the longer solo keeps an overlap, the other is trimmed around it
        kept = []
        for a, b, v, c in sorted(merged, key=lambda s: (-(s[1] - s[0]), s[0], s[2])):
            pieces = [(a, b)]
            for x, y, w, _ in kept:
                if w == v:
                    continue
                pieces = [p for s0, e0 in pieces
                          for p in ((s0, min(e0, x)), (max(s0, y), e0)) if p[1] > p[0]]
            kept += [(s0, e0, v, c) for s0, e0 in pieces
                     if (e0 - s0) * hop >= SOLO_MIN_S - 1e-9]
        return sorted(({"voice": v, "player": (player_of or {}).get(v), "start": _sec(a, hop),
                        "end": _sec(b, hop), "confidence": c} for a, b, v, c in kept),
                      key=lambda s: (s["start"], s["voice"]))
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


# A line sung TOGETHER (a group chant): another mic is singing (activity >= CHANT_ON, on its own
# scale) through CHANT_FRAC of the line, and is at least CHANT_REL as loud as the loudest one
# (so a neighbour's bleed into a mic never counts). With the mics' raw levels (`mic_raw`) it must
# also be really sung INTO that mic (Stefan 2026-11-11: one singer belting showed up as a chant,
# her bleed on the next mic): its raw level over the line within CHANT_OWN_DB of its own usual sung
# level (its median over the lines it leads), or, for a mic that leads no line, within CHANT_PEER_DB
# of the leading mic's raw level.
CHANT_ON = 0.3
CHANT_FRAC = 0.6
CHANT_REL = 0.45
CHANT_OWN_DB = 6.0
CHANT_PEER_DB = 6.0
# Per mic pair (Stefan 2026-11-15, Beautiful Chicken: 100 of 103 lines came up as chants). A mic's
# bleed from a given lead sits at a steady level under it; it only sings along when it's clearly
# above that. With CHANT_PAIR_MIN+ lines led by that lead, the other mic's usual bleed is the
# CHANT_BLEED_PCT percentile of (its level - the lead's) over them, and it counts only at
# CHANT_OVER_DB above that. The own-usual rule needs CHANT_OWN_MIN lines led (2-3 flukes led by a
# mic that mostly hears bleed made its "usual" a bleed level).
CHANT_PAIR_MIN = 6
CHANT_BLEED_PCT = 25
CHANT_OVER_DB = 8.0
CHANT_OWN_MIN = 5


def line_singers(lines, mic_act, mic_player=None, hop=HOP, mic_raw=None):
    """For each lyric line (song seconds), the vocal mic that is loudest over it (each mic judged on
    its own scale) -> [{start, end, stem, player, with}]. `mic_act` = {stem: activity}. `with` lists
    the other mics singing the same line with it (a group chant), loudest first; [] for a solo line.
    `mic_raw` = {stem: raw level} (optional): the bleed check above."""
    def mean(arr, a, b):
        x = np.asarray(arr, dtype=float)[a:b]
        return float(np.mean(x)) if len(x) else 0.0

    rows = []
    for s, e in lines or []:
        a, b = max(0, int(s / hop)), max(1, int(np.ceil(e / hop)))
        vals = {}
        for stem in sorted(mic_act):
            x = np.asarray(mic_act[stem], dtype=float)[a:b]
            vals[stem] = (float(np.mean(x)) if len(x) else 0.0,
                          float(np.mean(x >= CHANT_ON)) if len(x) else 0.0)
        best, best_v = None, -1.0
        for stem in sorted(vals):
            if vals[stem][0] > best_v:
                best, best_v = stem, vals[stem][0]
        rows.append((s, e, a, b, vals, best, best_v))
    own = {}
    if mic_raw:
        led = {}
        for s, e, a, b, vals, best, _ in rows:
            if best in mic_raw:
                led.setdefault(best, []).append(mean(mic_raw[best], a, b))
        own = {st: float(np.median(v)) for st, v in led.items() if len(v) >= CHANT_OWN_MIN}
    own_k, peer_k = 10 ** (-CHANT_OWN_DB / 20.0), 10 ** (-CHANT_PEER_DB / 20.0)
    db = lambda x: 20.0 * np.log10(max(float(x), 1e-12))
    floor = {}                                   # (lead, other) -> usual bleed, dB under the lead
    if mic_raw:
        diffs = {}
        for s, e, a, b, vals, best, _ in rows:
            if best not in mic_raw:
                continue
            lead_db = db(mean(mic_raw[best], a, b))
            for st in mic_raw:
                if st != best:
                    diffs.setdefault((best, st), []).append(db(mean(mic_raw[st], a, b)) - lead_db)
        floor = {k: float(np.percentile(v, CHANT_BLEED_PCT)) for k, v in diffs.items()
                 if len(v) >= CHANT_PAIR_MIN}

    def sung_into(st, a, b, best):
        if not mic_raw or st not in mic_raw:
            return True
        if (best, st) in floor:
            d = db(mean(mic_raw[st], a, b)) - db(mean(mic_raw[best], a, b))
            return d >= floor[(best, st)] + CHANT_OVER_DB
        r = mean(mic_raw[st], a, b)
        if st in own:
            return r >= own[st] * own_k
        return best in mic_raw and r >= mean(mic_raw[best], a, b) * peer_k

    out = []
    for s, e, a, b, vals, best, best_v in rows:
        together = []
        if best is not None and vals[best][1] >= CHANT_FRAC:
            # with raw levels the bleed test is sung_into (on each mic's own scale a mostly-bleed
            # mic always looks loud, so CHANT_REL would miss real duets next to it)
            together = sorted((st for st, (m, on) in vals.items() if st != best
                               and on >= CHANT_FRAC
                               and (sung_into(st, a, b, best) if mic_raw and st in mic_raw
                                    else m >= CHANT_REL * best_v)),
                              key=lambda st: (-vals[st][0], st))
        out.append({"start": round(float(s), 2), "end": round(float(e), 2), "stem": best,
                    "player": (mic_player or {}).get(best), "with": together,
                    "bleed_checked": bool(mic_raw)})
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


# ---- the ending: the last chord ringing out ------------------------------------------------------
RING_LOOK_S = 20.0        # the last chord is looked for in the song's final this-many seconds
RING_LOUD = 0.5           # "the band is loud": this fraction of its 90th-percentile level
RING_DROP_DB = 30.0       # the ring is over when the band is this far under the last hit...
RING_FLOOR_X = 2.0        # ...or within this many times its quietest second in the song...
RING_HOLD_S = 1.0         # ...for this long
RING_MAX_S = 10.0         # a ring-out is at most this long
RING_MIN_S = 1.5          # shorter than this isn't a ring-out (the band just stopped)
RING_PAST_S = 8.0         # the ring may run on this long past the song's labelled end


def ending(band, hop=HOP, song_n=None):
    """The song's last chord and where its ring dies away, from the band's summed level (close
    mics, so the crowd hardly counts): {last_hit, ring_end, confidence} in song seconds, or None
    when the band is still playing at the end, or just stops (no ring). `band` may run on past
    the song's labelled end (`song_n` frames, default all of it): the last chord must start
    inside the song, its ring may run on into the extra."""
    b = np.asarray(band, dtype=float)
    n = len(b)
    song_n = min(n, int(song_n or n))
    if song_n < int(5.0 / hop):
        return None
    b = _smooth(b, max(1, int(round(0.15 / hop))))
    ref = float(np.percentile(b[:song_n], 90)) + 1e-12
    tail = int(round(0.5 / hop))
    if float(b[-tail:].mean()) >= RING_LOUD * ref:
        return None                                          # still playing at the very end
    lo = max(0, song_n - int(round(RING_LOOK_S / hop)))
    loud = np.nonzero(b[lo:song_n + int(round(0.5 / hop))] >= RING_LOUD * ref)[0]
    if not len(loud):
        return None
    last_loud = lo + int(loud[-1])
    w0 = max(0, last_loud - int(round(1.0 / hop)))
    hit = w0 + int(np.argmax(b[w0:last_loud + 1]))
    peak = float(b[hit])
    sec = max(1, int(round(1.0 / hop)))
    # the floor: the quietest second anywhere (the gap after the song, when it's there)
    quiet = float(np.min(np.convolve(b, np.ones(sec) / sec, mode="valid"))) if n > sec else 0.0
    thr = max(peak * 10 ** (-RING_DROP_DB / 20.0), quiet * RING_FLOOR_X)
    hold = max(1, int(round(RING_HOLD_S / hop)))
    cap = min(n, hit + int(round(RING_MAX_S / hop)))
    end, i = None, hit + 1
    while i < cap:
        if b[i] < thr:
            j = i
            while j < min(n, i + hold) and b[j] < thr:
                j += 1
            if j - i >= hold or j >= n:
                end = i
                break
            i = j
        i += 1
    reached = end is not None
    if end is None:
        end = cap - 1
    if (end - hit) * hop < RING_MIN_S:
        return None
    drop = 20 * np.log10(peak / max(float(b[min(end, n - 1)]), 1e-12))
    conf = min(1.0, max(0.0, drop / RING_DROP_DB)) * (1.0 if reached else 0.6)
    return {"last_hit": _sec(hit, hop), "ring_end": _sec(end, hop), "confidence": round(conf, 2)}
