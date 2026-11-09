"""Who's-playing core (studio mode): deterministic per-stem RMS envelopes -> who is playing, who
leads, when. No AI. Moved here from movie_maker's fcpkit/whoplays.py with identical numbers, so any
app can read a multitrack the same way.

The caller supplies the show: `voices` ({VOICE: [stem files]}), `stem_dir`, the per-song wav or the
full-show (north-star) mix, and optional `player` ({stem file: name}). How stem files are grouped
into voices is the app's business (movie_maker reads its Ableton set); only three voice names are
special here: VOCAL (hysteresis, trumps all), BASS and DRUMS (rhythm rules). Every other voice is
a melodic lead candidate, treated identically.

Pipeline (plain numpy + ffmpeg):
  1. per-stem RMS energy envelope over the whole show (cached; tiny),
  2. locate the song: cross-correlate a per-song wav against the stem mix (analyze_song) or slice
     the show at a window of the north-star recording (analyze_window),
  3. normalize each voice's activity to its own featured level over the show,
  4. segment into stable "lead" spans (which voice is out front), with player and hand-offs.
"""

import hashlib
import os
import subprocess

import numpy as np

FFMPEG = os.environ.get("MUSIC_DETECTOR_FFMPEG", "/opt/homebrew/bin/ffmpeg")

RATE = 2000      # low rate is plenty: we need loudness-over-time of isolated stems, not pitch
HOP = 0.05       # 50 ms envelope frames (20 Hz)


# Tuning (deterministic thresholds; all easy to adjust during Stefan's validation).
# Activity is scale-invariant: each stem is normalized by ITS OWN "featured level"
# (a high percentile over the whole show), so Ableton's arbitrary per-stem gains don't
# matter — we ask "how hard is this player going, relative to their own loudest moments."
PCT_HI = 92          # per-stem featured-level percentile over the whole show
PRESENT = 0.30       # activity (fraction of own featured level) to count as playing/leading
# Vocals trump everything when present (you cut to the singer). Hysteresis: turn vocal ON
# at VOCAL_ON, keep it on through brief dips until it stays under VOCAL_OFF for VOCAL_OFF_S.
VOCAL_ON = 0.32      # bleed sits ~0.15, real singing ~0.4+, so this is comfortably between
VOCAL_ON_S = 0.2     # tiny debounce against single-frame flicker
VOCAL_OFF = 0.22
VOCAL_OFF_S = 0.6
# Reject vocal-on as bleed when a melodic instrument is much louder than the vocal (a blaring
# sax bleeds into the vocal mic). Singing isn't drowned like that; a lead vocal stays within
# ~this ratio of the loudest horn/guitar.
VOCAL_BLEED_RATIO = 1.8
# Drum solo overrides melodic *bleed*: drums clearly on top (not "everything else silent").
DRUM_DOM = 0.50      # drums at least this active ...
DRUM_MARGIN = 1.3    # ... and at least this many times the loudest other voice
LOW = 0.18           # below this a voice is treated as "off"
MIN_SPAN_S = 2.0     # merge lead spans shorter than this into a neighbor
SMOOTH_S = 1.0       # median-smooth the per-frame lead label over this window
ACT_CHUNK_S = 2.0    # a lead span longer than this is cut into ~this-long pieces, each with its OWN
                     # activity, so a song one voice leads throughout (a held vocal) still says who
                     # is playing WHEN (Black Heart was one 195 s span: every shot saw one average mix)
TRADE_MAX_S = 8.0    # alternating two leads, each span <= this -> a trading section


# ---------- decode + envelope (cached) ----------

def _decode(path, rate=RATE):
    p = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
        check=True, capture_output=True)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)


def _env_of(samples, rate=RATE, hop=HOP):
    frame = max(1, int(round(rate * hop)))
    nf = len(samples) // frame
    if nf == 0:
        return np.zeros(1)
    s = samples[:nf * frame].reshape(nf, frame)
    return np.sqrt((s * s).mean(axis=1))


def stem_env(path, cache_dir, rate=RATE, hop=HOP):
    """RMS envelope of a file, cached to disk (keyed by path+mtime+rate+hop)."""
    if cache_dir:
        st = os.stat(path)
        key = hashlib.sha1(
            f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}|{rate}|{hop}".encode()
        ).hexdigest()[:16]
        cache = os.path.join(cache_dir, f"env_{rate}_{int(hop*1000)}ms_{key}.npy")
        if os.path.exists(cache):
            return np.load(cache)
    e = _env_of(_decode(path, rate), rate, hop)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache, e)
    return e



def _energy_sum(files, cache_dir, stem_dir):
    """sqrt(sum of squared envelopes) over a list of stem files in `stem_dir` (energy adds)."""
    acc = None
    for f in files:
        e = stem_env(os.path.join(stem_dir, f), cache_dir)
        e2 = e * e
        if acc is None:
            acc = e2
        else:
            n = min(len(acc), len(e2))
            acc = acc[:n] + e2[:n]
    return np.sqrt(acc)


# ---------- alignment ----------

def _align(template, long, hop=HOP):
    """Lag (seconds) that best places `template` envelope inside `long` envelope,
    by normalized cross-correlation. Returns (lag_s, lag_frames, score)."""
    t = template - template.mean()
    norm = np.linalg.norm(t)
    if norm > 0:
        t = t / norm
    corr = np.correlate(long - long.mean(), t, mode="valid")
    lag = int(np.argmax(corr))
    return lag * hop, lag, float(corr[lag])


# ---------- analysis ----------

def _median_smooth(labels, win):
    """Mode filter over a window of frames (keeps labels categorical). A tie keeps the frame's own
    label if it is one of the tied, else the tied label seen first in the window — never set order,
    which changes from run to run with Python's string hashing (two leads alternating used to come
    out differently each time)."""
    if win <= 1:
        return labels
    out = list(labels)
    half = win // 2
    for i in range(len(labels)):
        a, b = max(0, i - half), min(len(labels), i + half + 1)
        window = labels[a:b]
        counts = {}
        for lab in window:
            counts[lab] = counts.get(lab, 0) + 1
        top = max(counts.values())
        tied = [lab for lab in counts if counts[lab] == top]     # dict keeps first-seen order
        out[i] = labels[i] if labels[i] in tied else tied[0]
    return out



def analyze_song(song_file, cache_dir, *, song_dir, stem_dir, voices, full_show=None, player=None):
    """Who's-playing timeline for one per-song wav (`song_dir`/`song_file`), located inside the
    whole-show stems by cross-correlating its envelope against the stem mix. Also finds the song's
    start in the full-show (north-star) mix when one is given, for later authoring."""
    song_path = os.path.join(song_dir, song_file)
    song_env = stem_env(song_path, cache_dir)        # envelope of the true-north song cut
    n = len(song_env)

    # whole-show per-voice energy (only for locating the song via alignment)
    voice_full = {v: _energy_sum(files, cache_dir, stem_dir) for v, files in voices.items()}
    L = min(len(e) for e in voice_full.values())
    mix = np.sqrt(sum((voice_full[v][:L] ** 2) for v in voice_full))

    lag_s, lag, score = _align(song_env, mix)
    # song start in true-north (Full Show) too, for authoring-time mapping later
    try:
        tn_env = stem_env(full_show, cache_dir)
        tn_lag_s, _, tn_score = _align(song_env, tn_env)
    except Exception:
        tn_lag_s, tn_score = None, None

    return _timeline_core(lag, n, cache_dir, voices=voices, stem_dir=stem_dir, player=player,
                          voice_full=voice_full, song=song_file,
                          stem_offset_s=round(lag_s, 2), stem_align_score=round(score, 1),
                          truenorth_offset_s=(None if tn_lag_s is None else round(tn_lag_s, 2)))


def _timeline_core(lag, n, cache_dir, *, voices, stem_dir, voice_full, player=None,
                   song=None, stem_offset_s=None, stem_align_score=None,
                   truenorth_offset_s=None):
    """Shared who's-playing detection over a window [lag, lag+n) of the whole-show per-voice
    energy envelopes. Both the per-song-wav path (analyze_song) and the single-file window
    path (analyze_window) feed it the same way; only how lag/n are found differs. No AI."""
    def win(e):
        seg = e[lag:lag + n]
        if len(seg) < n:                      # pad if song runs past cached mix end
            seg = np.concatenate([seg, np.zeros(n - len(seg))])
        return seg

    # Per-voice loudness, self-normalized to that voice's OWN featured level over the whole
    # show. Each voice's mics are combined by energy sum first (so multi-mic voices like
    # drums/vocals aren't inflated by a max over bleedy mics), then divided by the voice's
    # 92nd-pct energy over the show. This is scale-invariant per voice (Ableton gains
    # cancel), and cross-voice comparison is each voice relative to its own range.
    stem_full = {}
    for files in voices.values():
        for f in files:
            stem_full[f] = stem_env(os.path.join(stem_dir, f), cache_dir)
    phi_v = {v: (np.percentile(voice_full[v], PCT_HI) + 1e-9) for v in voices}
    act = {v: win(voice_full[v]) / phi_v[v] for v in voices}
    # Only three categories behave differently, so only these are referenced by name; give one the
    # show lacks a silent track rather than KeyError (an absent voice stays below every threshold).
    for _v in ("VOCAL", "BASS", "DRUMS"):
        act.setdefault(_v, np.zeros(n))
    # INSTRUMENT-AGNOSTIC: the melodic lead candidates are EVERY voice that isn't the vocal (its
    # own hysteresis path) or the bass/drums rhythm section — no hard-coded instrument list. They
    # all compete the same way (loudest wins); fair-share at build time spreads the spotlight, so
    # two instruments co-leading no longer needs a special "section" rule. Order = stem file order
    # (deterministic), which is the tie-break when two are exactly equal.
    melodic_voices = [v for v in act if v not in ("VOCAL", "BASS", "DRUMS")]

    # per-member activity (self-normalized) only for the "which player" detail
    memb = {}
    for v, files in voices.items():
        stack = np.stack([win(stem_full[f]) / (np.percentile(stem_full[f], PCT_HI) + 1e-9)
                          for f in files])
        memb[v] = (files, stack)

    # vocal hysteresis: "vox trumps everything" and holds through brief dips. Turns on only
    # when the vocal is up AND not drowned by a much-louder melodic instrument (bleed).
    melodic_max_arr = (np.maximum.reduce([act[v] for v in melodic_voices])
                       if melodic_voices else np.zeros(n))
    vocal_on = np.zeros(n, dtype=bool)
    on_hold = int(round(VOCAL_ON_S / HOP))
    off_hold = int(round(VOCAL_OFF_S / HOP))
    state, above, below = False, 0, 0
    for i in range(n):
        v = act["VOCAL"][i]
        can_on = (v >= VOCAL_ON and melodic_max_arr[i] <= VOCAL_BLEED_RATIO * v)
        if not state:
            above = above + 1 if can_on else 0
            if above >= on_hold:                       # confirmed singing; backfill the run
                state, below = True, 0
                vocal_on[max(0, i - above + 1):i + 1] = True
        else:
            vocal_on[i] = True
            below = below + 1 if v < VOCAL_OFF else 0
            if below >= off_hold:                      # phrase ended; trim trailing dip
                vocal_on[i - below + 1:i + 1] = False
                state, above = False, 0

    # per-frame lead: vocal (hysteresis) trumps all; then a drum solo over bleed; then the loudest
    # active melodic instrument (any of them, treated identically); then Stefan's rhythm rules.
    labels = []
    drum_dom = np.zeros(n, dtype=bool)
    bassy = np.zeros(n, dtype=bool)
    for i in range(n):
        aV = act["VOCAL"][i]
        aB, aD = act["BASS"][i], act["DRUMS"][i]
        mel = {v: act[v][i] for v in melodic_voices}
        melodic_max = max(mel.values()) if mel else 0.0
        other_max = max(aV, aB, melodic_max)
        drum_dom[i] = (aD >= DRUM_DOM and aD >= DRUM_MARGIN * other_max)
        bassy[i] = (aB >= PRESENT and melodic_max < LOW and aV < LOW)
        # a drum solo overrides a *held* vocal only when no one is actually singing now
        if drum_dom[i] and aV < VOCAL_ON:
            lead = "DRUMS"
        elif vocal_on[i]:
            lead = "VOCAL"
        elif drum_dom[i]:
            lead = "DRUMS"
        elif melodic_max >= PRESENT:
            lead = max((k for k in mel if mel[k] >= PRESENT), key=lambda k: mel[k])
        elif aD >= PRESENT and aB < LOW and melodic_max < LOW:
            lead = "DRUMS"
        elif aB >= PRESENT and melodic_max < LOW:
            lead = "BASS"           # bass is featured only when nothing melodic is going
        else:
            lead = "BAND"
        labels.append(lead)

    labels = _median_smooth(labels, int(round(SMOOTH_S / HOP)) | 1)

    # collapse to spans, then merge spans shorter than MIN_SPAN_S into a neighbor
    spans = []
    for i, lab in enumerate(labels):
        if spans and spans[-1][2] == lab:
            spans[-1][1] = i + 1
        else:
            spans.append([i, i + 1, lab])
    min_f = int(round(MIN_SPAN_S / HOP))
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for k in range(len(spans)):
            s, e, lab = spans[k]
            if e - s < min_f:
                # merge into the longer adjacent span
                left = spans[k - 1] if k > 0 else None
                right = spans[k + 1] if k < len(spans) - 1 else None
                if left and (not right or (left[1] - left[0]) >= (right[1] - right[0])):
                    left[1] = e
                elif right:
                    right[0] = s
                spans.pop(k)
                changed = True
                break
    # re-collapse adjacent same-label spans created by merging
    merged = []
    for s, e, lab in spans:
        if merged and merged[-1][2] == lab:
            merged[-1][1] = e
        else:
            merged.append([s, e, lab])
    spans = merged

    # which specific player leads a melodic/vocal voice over a span (scale-fair: argmax
    # of the per-member normalized activity, not raw level)
    players = player if player is not None else {}
    def who(voice_name, s, e):
        files, stack = memb[voice_name]
        return players.get(files[int(stack[:, s:e].mean(axis=1).argmax())], None)

    def longest_run(mask):
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best
    alone_min = int(round(1.0 / HOP))   # a sustained drum-dominant run this long = a solo/fill

    # trading section: a run of >=3 short, alternating spans between exactly two leads
    # (e.g. sax 6s / guitar 6s / sax 6s ...). Keep the sub-spans; just tag the group.
    trading = {}
    MEL = {"SAX", "TRUMPET", "GUITAR", "HORNS", "VOCAL"}
    i = 0
    while i < len(spans):
        j, labs = i, []
        while (j < len(spans) and spans[j][2] in MEL
               and (spans[j][1] - spans[j][0]) * HOP <= TRADE_MAX_S
               and len(set(labs + [spans[j][2]])) <= 2):
            labs.append(spans[j][2])
            j += 1
        if j - i >= 3 and len(set(labs)) == 2:
            pair = "↔".join(sorted(set(labs)))
            for k in range(i, j):
                trading[k] = pair
            i = j
        else:
            i += 1

    out_spans = []
    chunk_f = max(1, int(round(ACT_CHUNK_S / HOP)))
    for idx, (s0, e0, lab) in enumerate(spans):
        solo = None                                  # judged on the WHOLE lead span
        if lab == "DRUMS" and longest_run(drum_dom[s0:e0]) >= alone_min:
            solo = "drum solo/fill"
        if lab == "BASS" and (e0 - s0) * HOP >= 3.0:
            solo = "bass solo"
        k = max(1, int(round((e0 - s0) / chunk_f)))
        for j in range(k):
            s = s0 + (e0 - s0) * j // k
            e = s0 + (e0 - s0) * (j + 1) // k
            detail = {"start_s": round(s * HOP, 2), "end_s": round(e * HOP, 2),
                      "dur_s": round((e - s) * HOP, 2), "lead": lab}
            if lab in ("SAX", "TRUMPET", "GUITAR", "VOCAL"):
                detail["player"] = who(lab, s, e)
            elif lab == "HORNS":
                detail["player"] = f"{who('SAX', s, e)} (sax) + {who('TRUMPET', s, e)} (tpt)"
            if solo:
                detail["solo"] = solo
            if idx in trading:
                detail["trading"] = trading[idx]
            detail["activity"] = dict(sorted(
                {v: round(float(act[v][s:e].mean()), 2) for v in voices}.items(),
                key=lambda kv: -kv[1]))
            out_spans.append(detail)

    return {
        "song": song,
        "song_len_s": round(n * HOP, 2),
        "stem_offset_s": stem_offset_s,
        "stem_align_score": stem_align_score,
        "truenorth_offset_s": truenorth_offset_s,
        "rate": RATE, "hop_s": HOP,
        "spans": out_spans,
    }



def analyze_window(start_s, end_s, cache_dir, *, stem_dir, voices, full_show,
                   song=None, player=None):
    """Who's-playing timeline for a song that is a WINDOW [start_s, end_s] (seconds) into the
    show's one continuous north-star recording -- the common case, where there is NO separate
    per-song wav. Instead of locating a wav by cross-correlation, it slices the whole-show
    stems at the window. The stems and the north-star mix can start at different times (Co-Op's
    differ by ~31 s), so it aligns the north-star mix to the stem mix ONCE and shifts the window
    by that global offset. start_s/end_s are in NORTH-STAR time, so truenorth_offset_s = start_s
    (what the authoring time map wants). Same detection as analyze_song. Deterministic, no AI."""
    voice_full = {v: _energy_sum(files, cache_dir, stem_dir) for v, files in voices.items()}
    L = min(len(e) for e in voice_full.values())
    stem_mix = np.sqrt(sum((voice_full[v][:L] ** 2) for v in voice_full))
    # Where north-star t=0 sits in stem time (constant global offset; both are continuous).
    g_lag_s = 0.0
    try:
        ns_env = stem_env(full_show, cache_dir)
        g_lag_s, _, _ = _align(ns_env, stem_mix)
    except Exception:
        g_lag_s = 0.0
    lag = max(0, int(round((start_s + g_lag_s) / HOP)))
    n = max(1, int(round((end_s - start_s) / HOP)))
    return _timeline_core(lag, n, cache_dir, voices=voices, stem_dir=stem_dir, player=player,
                          voice_full=voice_full, song=song,
                          stem_offset_s=round(lag * HOP, 2), stem_align_score=None,
                          truenorth_offset_s=round(start_s, 2))


def analyze_parts(parts_dir, pad_s, song_len_s, cache_dir, *, voices, song=None, player=None,
                  truenorth_offset_s=None):
    """Who's-playing timeline for ONE song's parts split out of a mix (music_detector.separate):
    every file in `parts_dir` starts `pad_s` seconds before the song, so the song is the window
    [pad_s, pad_s + song_len_s) of them -- no alignment needed. Same detection as analyze_window."""
    voice_full = {v: _energy_sum(files, cache_dir, parts_dir) for v, files in voices.items()}
    lag = max(0, int(round(pad_s / HOP)))
    n = max(1, int(round(song_len_s / HOP)))
    return _timeline_core(lag, n, cache_dir, voices=voices, stem_dir=parts_dir, player=player,
                          voice_full=voice_full, song=song, stem_offset_s=round(lag * HOP, 2),
                          stem_align_score=None, truenorth_offset_s=truenorth_offset_s)


def report(tl):
    """Readable summary Stefan can check against the song by ear."""
    def mmss(s):
        return f"{int(s)//60}:{int(s)%60:02d}"
    lines = [
        f"WHO'S PLAYING — {tl['song']}  ({mmss(tl['song_len_s'])}, "
        f"stem offset {mmss(tl['stem_offset_s'])}, align score {tl['stem_align_score']})",
        "-" * 64,
    ]
    for sp in tl["spans"]:
        who = f"  [{sp['player']}]" if sp.get("player") else ""
        tag = ""
        if sp.get("solo"):
            tag += f"  <{sp['solo']}>"
        if sp.get("trading"):
            tag += f"  <trading {sp['trading']}>"
        top = " ".join(f"{k}:{v}" for k, v in list(sp["activity"].items())[:3])
        lines.append(f"  {mmss(sp['start_s'])}–{mmss(sp['end_s'])}  "
                     f"({sp['dur_s']:>4.1f}s)  {sp['lead']:<7}{who:<22}{tag:<22}  {top}")
    return "\n".join(lines)
