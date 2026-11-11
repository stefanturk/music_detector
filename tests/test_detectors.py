"""Each detector on synthetic activity curves (50 ms frames)."""
import numpy as np

from music_detector.core import detectors as D

HOP = D.HOP


def frames(s):
    return int(round(s / HOP))


def steady(sec, level=1.0):
    return np.full(frames(sec), float(level))


def test_playing_turns_on_and_off_with_hysteresis():
    x = np.concatenate([steady(2, 0.0), steady(5, 0.8), steady(0.5, 0.1), steady(3, 0.8),
                        steady(3, 0.0)])
    m = D.playing_mask(x)
    rs = D.runs(m)
    assert len(rs) == 1                                   # a 0.5 s dip doesn't stop it
    a, b = rs[0]
    assert abs(a * HOP - 2.0) < 0.06 and abs(b * HOP - 10.5) < 0.06


def test_a_tom_burst_is_a_fill_and_kick_snare_alone_is_not():
    n = frames(30)
    t = np.arange(n) * HOP
    ks = ((t * 2) % 1.0 < 0.1).astype(float)              # kick/snare every half second
    toms = np.full(n, 0.02)
    burst = (t >= 12.0) & (t < 13.5)
    toms[burst] = 1.0
    fills = D.drum_fills(toms=toms, kick_snare=ks)
    assert len(fills) == 1
    f = fills[0]
    assert abs(f["start"] - 12.0) < 0.1 and abs(f["end"] - 13.5) < 0.1
    assert f["confidence"] == "high" and f["strength"] >= D.FILL_BIG
    assert D.drum_fills(toms=np.full(n, 0.02) + 0.01 * ks, kick_snare=ks) == []


def test_without_tom_mics_a_drum_burst_is_a_low_confidence_fill():
    n = frames(40)
    t = np.arange(n) * HOP
    d = 0.5 + 0.5 * ((t % 0.5) < 0.05)                    # steady groove: a hit every 0.5 s
    burst = (t >= 20) & (t < 21.5)
    d[burst] = 1.0 + 1.5 * ((t[burst] % 0.1) < 0.05)      # fast, loud hits
    fills = D.drum_fills(drums=d)
    assert fills and all(f["confidence"] == "low" for f in fills)
    assert any(f["start"] <= 21.0 and f["end"] >= 20.5 for f in fills)


def test_keys_score_when_the_guitar_and_horns_drop_out():
    # keys at the same steady level all song; guitar + sax drop out for 10 s in the middle
    keys = steady(40, 0.8)
    gtr = np.concatenate([steady(15, 1.0), steady(10, 0.0), steady(15, 1.0)])
    sax = gtr.copy()
    act = {"KEYS": keys, "GUITAR": gtr, "SAX": sax}
    playing = {v: D.playing_mask(a) for v, a in act.items()}
    f = D.featured_scores(act, playing)["KEYS"]
    bins = lambda a, b: f[int(a / D.BIN_S):int(b / D.BIN_S)]
    assert bins(17, 23).mean() > 0.4 and bins(2, 12).mean() < 0.1


def test_a_solo_while_nobody_sings():
    gtr = np.concatenate([steady(10, 0.4), steady(8, 1.2), steady(10, 0.4)])
    sax = steady(28, 0.3)
    vocal = np.concatenate([steady(10, 0.8), steady(8, 0.0), steady(10, 0.8)])
    act = {"GUITAR": gtr, "SAX": sax, "VOCAL": vocal, "DRUMS": steady(28, 1.0),
           "BASS": steady(28, 1.0)}
    playing = {v: D.playing_mask(a) for v, a in act.items()}
    sing = D.singing_mask(len(gtr), lines=[(0.5, 9.8), (18.2, 27.5)])
    sol = D.solos(act, playing, sing, {"GUITAR": "Stefan"})
    assert len(sol) == 1 and sol[0]["voice"] == "GUITAR" and sol[0]["player"] == "Stefan"
    assert abs(sol[0]["start"] - 10.0) < 0.3 and abs(sol[0]["end"] - 18.0) < 0.3
    # the same guitar line UNDER singing is not a solo
    assert D.solos(act, playing, np.ones(len(gtr), dtype=bool)) == []


def test_an_entrance_onset_to_50_ms():
    x = np.concatenate([steady(5, 0.0), np.linspace(0.05, 0.9, 4), steady(5, 0.9)])
    act = {"SAX": x}
    ent = D.entrances(act, {"SAX": D.playing_mask(x)})
    assert len(ent) == 1 and abs(ent[0]["at"] - 5.0) <= 0.05


def test_energy_rises_across_a_build():
    n = frames(60)
    ramp = np.linspace(0.1, 1.5, n)
    e = D.energy_curve({"A": ramp, "B": ramp * 0.5})
    assert e[5] < 0.2 and e[-5] > 0.8
    assert np.all(np.diff(e[10:-10]) >= -0.05)            # it climbs (smoothed)


def test_each_lyric_line_goes_to_the_loudest_mic():
    left = np.concatenate([steady(4, 1.0), steady(4, 0.1)])
    right = np.concatenate([steady(4, 0.1), steady(4, 1.0)])
    out = D.line_singers([(0.5, 3.5), (4.5, 7.5)], {"11 VOX LEFT.wav": left,
                                                     "13 VOX RIGHT.wav": right},
                         {"11 VOX LEFT.wav": "Micah"})
    assert [(s["stem"], s["player"]) for s in out] == [("11 VOX LEFT.wav", "Micah"),
                                                       ("13 VOX RIGHT.wav", None)]


def test_a_line_sung_together_is_a_group_chant():
    left = np.concatenate([steady(4, 1.0), steady(4, 1.0)])
    right = np.concatenate([steady(4, 0.15), steady(4, 0.9)])       # bleed, then singing too
    out = D.line_singers([(0.5, 3.5), (4.5, 7.5)], {"11 VOX LEFT.wav": left,
                                                     "13 VOX RIGHT.wav": right})
    assert out[0]["with"] == [] and out[1]["with"] == ["13 VOX RIGHT.wav"]


def test_best_moments_find_the_featured_stretch():
    f = np.zeros(120)
    f[60:70] = 0.9
    best = D.best_moments({"KEYS": f})["KEYS"]
    assert best and 28.0 <= best[0]["start"] <= 31.0 and best[0]["score"] > 0.5


def test_trading_sections_merge_from_spans():
    spans = [{"start_s": 0.0, "end_s": 2.0, "trading": "SAX↔GUITAR"},
             {"start_s": 2.0, "end_s": 4.0, "trading": "SAX↔GUITAR"},
             {"start_s": 4.0, "end_s": 6.0}]
    assert D.trading(spans) == [{"pair": "SAX↔GUITAR", "start": 0.0, "end": 4.0}]


def test_median_smooth_ties_do_not_depend_on_hashing():
    import os
    import subprocess
    import sys
    code = ("from music_detector.core import whoplays as W;"
            "print(W._median_smooth(['SAX','GUITAR','KEYS']*10, 5))")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                           env=dict(os.environ, PYTHONHASHSEED=str(seed))).stdout
            for seed in (0, 1, 2, 3, 4)}
    assert len(outs) == 1 and outs.pop().strip()          # same answer whatever the hash seed
    from music_detector.core import whoplays as W
    labels = ["SAX", "GUITAR", "KEYS"] * 10
    a = W._median_smooth(labels, 5)
    assert a[4] == "KEYS"            # K S [G] K S: a 2-2 tie, the frame's own label isn't in it,
    assert W._median_smooth(["A", "B", "B", "A", "C"], 5)[2] == "B"   # so the first seen wins; own wins


def test_line_singers_needs_lines_and_mics():
    from music_detector.core import studio
    assert studio.line_singers(0.0, 10.0, "/nonexistent", stem_dir="/nonexistent", vocal_files=[],
                               lines=[(1.0, 2.0)]) == []
    assert studio.line_singers(0.0, 10.0, "/nonexistent", stem_dir="/nonexistent",
                               vocal_files=["a.wav"], lines=[]) == []


def _ring(decay_s, tail_s, floor=0.001):
    """A song at level 1 with a final hit at 1.5, its ring decaying exponentially -40 dB over
    decay_s, then the room floor."""
    song = steady(30.0, 1.0)
    hit = np.full(frames(0.2), 1.5)
    k = np.arange(frames(decay_s))
    ring = 1.5 * 10 ** (-2.0 * k / max(len(k), 1))
    return np.concatenate([song, hit, ring, np.full(frames(tail_s), floor)])


def test_the_last_chord_and_its_ring_out():
    b = _ring(4.0, 4.0)
    e = D.ending(b, song_n=frames(32.0))
    assert e and abs(e["last_hit"] - 30.0) <= 0.2
    # -30 dB of a -40 dB decay over 4 s is 3 s after the hit (plus the 0.2 s hit)
    assert abs(e["ring_end"] - 33.2) <= 0.3 and e["confidence"] >= 0.9


def test_a_band_that_stops_dead_has_no_ring():
    b = np.concatenate([steady(30.0, 1.0), np.full(frames(4.0), 0.001)])
    assert D.ending(b) is None


def test_a_band_still_playing_at_the_end_has_no_ring():
    assert D.ending(steady(40.0, 1.0)) is None


def test_the_last_chord_must_start_inside_the_song():
    # the next song starts loud right after a short gap: its chords aren't this song's ending
    b = np.concatenate([_ring(3.0, 1.0), steady(1.0, 0.0005), steady(3.0, 1.0), steady(1.0, 0.0005)])
    e = D.ending(b, song_n=frames(31.0))
    assert e is None or e["last_hit"] <= 31.5


# ---- solos judged on each player's own usual level and sound (lead=) ----
def _bins_of(sec_total, parts, default):
    """Per-BIN_S array: `default`, with `parts` [(a_s, b_s, value)] laid over it."""
    nb = int(round(sec_total / D.BIN_S))
    x = np.full(nb, float(default))
    for a, b, v in parts:
        x[int(a / D.BIN_S):int(b / D.BIN_S)] = v
    return x


def _band(gtr_up=True, keys_up=False):
    """60 s: the guitar sits at 0.5 and plays a lead line from 20 to 45 s; the keys comp at 0.7
    (louder than the guitar all along) with an answering phrase at 30-31.5 s."""
    gtr = np.concatenate([steady(20, 0.5), steady(25, 0.65 if gtr_up else 0.5), steady(15, 0.5)])
    keys = np.concatenate([steady(20, 0.7), steady(10, 0.9 if keys_up else 0.7),
                           steady(1.5, 1.0), steady(13.5, 0.9 if keys_up else 0.7), steady(15, 0.7)])
    act = {"GUITAR": gtr, "KEYS": keys, "DRUMS": steady(60, 1.0), "BASS": steady(60, 1.0),
           "VOCAL": steady(60, 0.0)}
    lead = {"GUITAR": {"clarity": _bins_of(60, [(20, 45, 0.92)], 0.6),
                       "vibrato": _bins_of(60, [(20, 45, 0.3)], 0.0),
                       "playing": _bins_of(60, [], 1.0)},
            "KEYS": {"clarity": _bins_of(60, [], 0.85), "vibrato": _bins_of(60, [], 0.0),
                     "playing": _bins_of(60, [], 1.0)}}
    playing = {v: D.playing_mask(a) for v, a in act.items()}
    sing = D.singing_mask(frames(60), lines=[(1.0, 18.0), (47.0, 59.0)])
    return act, playing, sing, lead


def test_a_lead_line_up_on_its_own_usual_is_a_solo_over_louder_comping():
    act, playing, sing, lead = _band()
    assert not [s for s in D.solos(act, playing, sing) if s["voice"] == "GUITAR"]   # loudness alone
    sol = D.solos(act, playing, sing, {"GUITAR": "Stefan"}, lead=lead)
    g = [s for s in sol if s["voice"] == "GUITAR"]
    assert len(g) == 1 and g[0]["player"] == "Stefan" and g[0]["confidence"] >= 0.3
    assert g[0]["start"] <= 21.5 and g[0]["end"] >= 43.5        # the keys' answer doesn't end it
    assert not [s for s in sol if s["voice"] == "KEYS"]


def test_louder_chords_are_not_a_solo():
    act, playing, sing, lead = _band(gtr_up=False, keys_up=True)
    act["GUITAR"] = act["GUITAR"] + 0.25          # the guitar comps along, so loudness alone is quiet
    playing = {v: D.playing_mask(a) for v, a in act.items()}
    assert not [s for s in D.solos(act, playing, sing) if s["voice"] in ("KEYS", "GUITAR")]
    lead["GUITAR"] = {"clarity": _bins_of(60, [], 0.6), "vibrato": _bins_of(60, [], 0.0),
                      "playing": _bins_of(60, [], 1.0)}
    sol = D.solos(act, playing, sing, lead=lead)
    assert not [s for s in sol if s["voice"] in ("KEYS", "GUITAR")]


def test_a_lead_line_under_singing_is_not_a_solo():
    act, playing, _sing, lead = _band()
    assert D.solos(act, playing, np.ones(frames(60), dtype=bool), lead=lead) == []


def test_without_the_sound_the_loudness_rule_is_unchanged():
    act, playing, sing, _lead = _band()
    assert D.solos(act, playing, sing, lead=None) == D.solos(act, playing, sing)
