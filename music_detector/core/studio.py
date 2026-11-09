"""Studio mode: run every detector on one song of a multitrack show -> one JSON-able dict.

    from music_detector.core import studio
    res = studio.detect(stem_offset_s, song_len_s, cache_dir, stem_dir=..., voices=..., player=...,
                        lines=[(start, end), ...], spans=whoplays_result["spans"])

`stem_offset_s` / `song_len_s` say where the song sits in the stems (whoplays reports both).
`voices` = {VOICE: [stem files]} as for whoplays; `player` = {stem file: name}. `lines` are the lyric
lines in song seconds (optional: without them, singing = the VOCAL voice playing). `spans` are the
song's who's-playing spans (optional, for trading sections).

Result schema (schema 1; times in song seconds):
  song, schema, hop_s, bin_s, stem_offset_s, song_len_s, voices: [VOICE...]
  energy:    [0..1 per bin]                       whole-band intensity
  featured:  {VOICE: [0..1 per bin]}              stands out vs its own usual share and level
  playing:   {VOICE: [[start, end], ...]}
  fills:     [{start, end, strength 0..1, toms 0..1|None, confidence "high"|"low"}]
  solos:     [{voice, player, start, end, confidence 0..1}]
  trading:   [{pair, start, end}]
  entrances: [{voice, at}]
  singers:   [{start, end, stem, player}]         one per lyric line (VOCAL mics)
  best:      {VOICE: [{start, end, score, player}]}
"""
import os
import re

import numpy as np

from music_detector.core import detectors as D
from music_detector.core import whoplays as W

SCHEMA = 1
_TOM = re.compile(r"\b(tom|toms|floor)\b", re.I)
_KS = re.compile(r"\b(kick|kik|bd|snare|snr|sd)\b", re.I)


def _tokens(fn):
    return " ".join(t for t in re.split(r"[\s_\-.]+", os.path.splitext(fn)[0]) if t)


def _window(e, lag, n):
    seg = e[lag:lag + n]
    if len(seg) < n:
        seg = np.concatenate([seg, np.zeros(n - len(seg))])
    return seg


def activity(stem_offset_s, song_len_s, cache_dir, *, stem_dir, voices):
    """Per-voice activity over the song, exactly as whoplays computes it (each voice divided by its
    92nd-percentile level over the whole show). Returns (act, lag, n, voice_full)."""
    voice_full = {v: W._energy_sum(files, cache_dir, stem_dir) for v, files in voices.items()}
    lag = max(0, int(round(stem_offset_s / W.HOP)))
    n = max(1, int(round(song_len_s / W.HOP)))
    act = {v: _window(voice_full[v], lag, n) / (np.percentile(voice_full[v], W.PCT_HI) + 1e-9)
           for v in voices}
    return act, lag, n, voice_full


def _sum_stems(files, cache_dir, stem_dir, lag, n):
    if not files:
        return None
    return _window(W._energy_sum(files, cache_dir, stem_dir), lag, n)


def _mic_act(files, cache_dir, stem_dir, lag, n):
    out = {}
    for f in files:
        e = W.stem_env(os.path.join(stem_dir, f), cache_dir)
        out[f] = _window(e, lag, n) / (np.percentile(e, W.PCT_HI) + 1e-9)
    return out


def detect(stem_offset_s, song_len_s, cache_dir, *, stem_dir, voices, player=None, lines=None,
           spans=None, song=None):
    player = player or {}
    act, lag, n, _ = activity(stem_offset_s, song_len_s, cache_dir, stem_dir=stem_dir, voices=voices)
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
    singers = (D.line_singers(lines, _mic_act(vox, cache_dir, stem_dir, lag, n), player)
               if lines and vox else [])
    return {
        "song": song, "schema": SCHEMA, "hop_s": D.HOP, "bin_s": D.BIN_S,
        "stem_offset_s": round(float(stem_offset_s), 2), "song_len_s": round(float(song_len_s), 2),
        "voices": sorted(act),
        "energy": [round(float(x), 3) for x in D.energy_curve(act)],
        "featured": {v: [round(float(x), 3) for x in f] for v, f in sorted(feat.items())},
        "playing": {v: [[D._sec(a), D._sec(b)] for a, b in D.runs(m)]
                    for v, m in sorted(playing.items())},
        "fills": fills,
        "solos": D.solos(act, playing, sing, player_of),
        "trading": D.trading(spans),
        "entrances": D.entrances(act, playing),
        "singers": singers,
        "best": D.best_moments(feat, player_of=player_of),
    }


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
