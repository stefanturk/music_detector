"""Studio mode: run every detector on one song of a multitrack show -> one JSON-able dict.

    from music_detector.core import studio
    res = studio.detect(stem_offset_s, song_len_s, cache_dir, stem_dir=..., voices=..., player=...,
                        lines=[(start, end), ...], spans=whoplays_result["spans"])

`stem_offset_s` / `song_len_s` say where the song sits in the stems (whoplays reports both).
`voices` = {VOICE: [stem files]} as for whoplays; `player` = {stem file: name}. `lines` are the lyric
lines in song seconds (optional: without them, singing = the VOCAL voice playing). `spans` are the
song's who's-playing spans (optional, for trading sections).

Result schema (schema 1; times in song seconds):
  song, schema, version, hop_s, bin_s, stem_offset_s, song_len_s, voices: [VOICE...]
  energy:    [0..1 per bin]                       whole-band intensity
  featured:  {VOICE: [0..1 per bin]}              stands out vs its own usual share and level
  playing:   {VOICE: [[start, end], ...]}
  fills:     [{start, end, strength 0..1, toms 0..1|None, confidence "high"|"low"}]
  solos:     [{voice, player, start, end, confidence 0..1}]
  trading:   [{pair, start, end}]
  entrances: [{voice, at}]
  singers:   [{start, end, stem, player}]         one per lyric line (VOCAL mics)
  best:      {VOICE: [{start, end, score, player}]}
  ending:    {last_hit, ring_end, confidence}      the last chord and where its ring dies
Every output is read by Movie Maker (flow.py / engine.py); the table is in the README.
"""
import os
import re

import numpy as np

from music_detector.core import detectors as D
from music_detector.core import whoplays as W

SCHEMA = 1
VERSION = 5              # 5: chants judged per mic pair against its usual bleed; 4: group chants checked for bleed (bleed_checked); 2: + ending (the last chord's ring-out); 3: solos judged on each
                         # player's own usual level and sound (lead_sound)
_TOM = re.compile(r"\b(tom|toms|floor)\b", re.I)
_KS = re.compile(r"\b(kick|kik|bd|snare|snr|sd)\b", re.I)


def _tokens(fn):
    return " ".join(t for t in re.split(r"[\s_\-.]+", os.path.splitext(fn)[0]) if t)


def _window(e, lag, n):
    seg = e[lag:lag + n]
    if len(seg) < n:
        seg = np.concatenate([seg, np.zeros(n - len(seg))])
    return seg


def activity(stem_offset_s, song_len_s, cache_dir, *, stem_dir, voices, bleed_guard=False):
    """Per-voice activity over the song, exactly as whoplays computes it (each voice divided by its
    92nd-percentile level over the whole show). Returns (act, lag, n, voice_full)."""
    voice_full = {v: W._energy_sum(files, cache_dir, stem_dir) for v, files in voices.items()}
    lag = max(0, int(round(stem_offset_s / W.HOP)))
    n = max(1, int(round(song_len_s / W.HOP)))
    act = {v: _window(voice_full[v], lag, n) / (np.percentile(voice_full[v], W.PCT_HI) + 1e-9)
           for v in voices}
    if bleed_guard:
        from music_detector.core import eq as _eq
        act = _eq.guard_activity(act, voices, stem_dir, lag, n, W.HOP)
    return act, lag, n, voice_full


def _sum_stems(files, cache_dir, stem_dir, lag, n):
    if not files:
        return None
    return _window(W._energy_sum(files, cache_dir, stem_dir), lag, n)


def _mic_act(files, cache_dir, stem_dir, lag, n, raw=None):
    """{file: activity} of each vocal mic (its level on its own scale); `raw` (a dict) also gets
    each mic's raw level over the song, for the chant bleed check."""
    out = {}
    for f in files:
        e = W.stem_env(os.path.join(stem_dir, f), cache_dir)
        w = _window(e, lag, n)
        out[f] = w / (np.percentile(e, W.PCT_HI) + 1e-9)
        if raw is not None:
            raw[f] = w
    return out


def lead_sound(stem_offset_s, song_len_s, cache_dir, *, stem_dir, voices, ffmpeg=None):
    """{voice: timbre.measure(...)} of each melodic voice's loudest stem over the song (BIN_S bins):
    how clear a single pitch it plays and how much it bends, for the own-usual solo rule. Voices
    whose audio can't be read are left out; None when nothing can be read (no ffmpeg)."""
    from music_detector.core import eq as _eq
    from music_detector.core import timbre as T
    ffmpeg = ffmpeg or os.environ.get("MUSIC_DETECTOR_FFMPEG", "/opt/homebrew/bin/ffmpeg")
    if not os.path.exists(ffmpeg):
        return None
    lag = max(0, int(round(stem_offset_s / W.HOP)))
    n = max(1, int(round(song_len_s / W.HOP)))
    nb = int(np.ceil(song_len_s / D.BIN_S))
    out = {}
    for v, files in (voices or {}).items():
        if v in D.RHYTHM or v == "VOCAL" or not files:
            continue
        try:
            loud = []
            for f in files:
                seg = W.stem_env(os.path.join(stem_dir, f), cache_dir)[lag:lag + n]
                loud.append((float(np.percentile(seg, 90)) if len(seg) else 0.0, f))
            best = max(loud)[1]
            x = _eq._decode(os.path.join(stem_dir, best), stem_offset_s, song_len_s, ffmpeg,
                            rate=T.RATE)
            if x is None or not len(x):
                continue
            m = T.measure(x, n_bins=nb)
        except Exception:
            continue
        out[v] = {k: m[k] for k in ("clarity", "vibrato", "playing")}
    return out or None


def detect(stem_offset_s, song_len_s, cache_dir, *, stem_dir, voices, player=None, lines=None,
           spans=None, song=None, bleed_guard=False, lead=True):
    """`lead`: True measures each melodic part's sound for the own-usual solo rule (lead_sound),
    False skips it (the loudness rule alone), or a ready {voice: measurements} dict."""
    player = player or {}
    act, lag, n, voice_full = activity(stem_offset_s, song_len_s, cache_dir, stem_dir=stem_dir,
                                       voices=voices, bleed_guard=bleed_guard)
    if lead is True:
        lead = lead_sound(stem_offset_s, song_len_s, cache_dir, stem_dir=stem_dir, voices=voices)
    playing = {v: D.playing_mask(a) for v, a in act.items()}
    sing = D.singing_mask(n, lines, act.get("VOCAL"))
    # a voice's player, when all its stems name the same person (SAX: Drew; VOCAL: several -> None)
    player_of = {}
    for v, files in voices.items():
        names = {player.get(f) for f in files} - {None}
        if len(names) == 1:
            player_of[v] = names.pop()

    drum_files = voices.get("DRUMS", [])
    toms = [f for f in drum_files if _TOM.search(_tokens(f))]
    ks = [f for f in drum_files if _KS.search(_tokens(f))]
    if toms:
        fills = D.drum_fills(toms=_sum_stems(toms, cache_dir, stem_dir, lag, n),
                             kick_snare=_sum_stems(ks, cache_dir, stem_dir, lag, n))
    else:
        fills = D.drum_fills(drums=act.get("DRUMS"))

    feat = D.featured_scores(act, playing)
    vox = voices.get("VOCAL", [])
    raw = {}
    singers = (D.line_singers(lines, _mic_act(vox, cache_dir, stem_dir, lag, n, raw), player,
                              mic_raw=raw)
               if lines and vox else [])
    return {
        "song": song, "schema": SCHEMA, "version": VERSION, "hop_s": D.HOP, "bin_s": D.BIN_S,
        "stem_offset_s": round(float(stem_offset_s), 2), "song_len_s": round(float(song_len_s), 2),
        "voices": sorted(act),
        "energy": [round(float(x), 3) for x in D.energy_curve(act)],
        "featured": {v: [round(float(x), 3) for x in f] for v, f in sorted(feat.items())},
        "playing": {v: [[D._sec(a), D._sec(b)] for a, b in D.runs(m)]
                    for v, m in sorted(playing.items())},
        "fills": fills,
        "solos": D.solos(act, playing, sing, player_of, lead=lead or None, featured=feat),
        "trading": D.trading(spans),
        "entrances": D.entrances(act, playing),
        "singers": singers,
        "best": D.best_moments(feat, player_of=player_of),
        "ending": D.ending(band_level(voice_full, lag, n + int(round(D.RING_PAST_S / D.HOP))),
                           song_n=n),
    }


def band_level(voice_full, lag, n):
    """The band's summed level over the song (every voice but the vocal mics: they hear the
    crowd and the talking)."""
    parts = [_window(e, lag, n) for v, e in voice_full.items() if v != "VOCAL"]
    if not parts:
        return np.zeros(n)
    return np.sqrt(sum(p ** 2 for p in parts))


def line_singers(stem_offset_s, song_len_s, cache_dir, *, stem_dir, vocal_files, player=None,
                 lines=None):
    """Who sings each lyric line: the loudest of the VOCAL mics over it -> [{start, end, stem,
    player}] (the `singers` part of detect, for when the lyrics are made after the analysis)."""
    if not (lines and vocal_files):
        return []
    lag = max(0, int(round(stem_offset_s / W.HOP)))
    n = max(1, int(round(song_len_s / W.HOP)))
    raw = {}
    act = _mic_act(vocal_files, cache_dir, stem_dir, lag, n, raw)
    return D.line_singers(lines, act, player or {}, mic_raw=raw)


def summary(res):
    """A short readable report (no lyrics): fills, solos, best moments, singers per person."""
    def mmss(s):
        return f"{int(s) // 60}:{int(s) % 60:02d}"
    out = [f"DETECTED — {res.get('song')}  ({mmss(res['song_len_s'])})"]
    out.append(f"fills: {len(res['fills'])}")
    for f in res["fills"]:
        out.append(f"  {mmss(f['start'])}  {f['end'] - f['start']:.1f}s  strength {f['strength']}"
                   f"  toms {f['toms']}  ({f['confidence']})")
    out.append(f"solos: {len(res['solos'])}")
    for s in res["solos"]:
        out.append(f"  {mmss(s['start'])}-{mmss(s['end'])}  {s['voice']} {s['player'] or ''}"
                   f"  conf {s['confidence']}")
    out.append(f"trading: {len(res['trading'])}")
    for t in res["trading"]:
        out.append(f"  {mmss(t['start'])}-{mmss(t['end'])}  {t['pair']}")
    out.append("best moments:")
    for v, picks in res["best"].items():
        out.append(f"  {v:8} " + ", ".join(f"{mmss(p['start'])} ({p['score']})" for p in picks))
    out.append(f"entrances: " + ", ".join(f"{e['voice']}@{mmss(e['at'])}" for e in res["entrances"]))
    if res["singers"]:
        cnt = {}
        for s in res["singers"]:
            k = s["player"] or s["stem"]
            cnt[k] = cnt.get(k, 0) + 1
        out.append("lines per singer: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    e = res["energy"]
    if e:
        out.append("energy (per 10 s): " + " ".join(
            f"{np.mean(e[i:i + 20]):.1f}" for i in range(0, len(e), 20)))
    return "\n".join(out)
