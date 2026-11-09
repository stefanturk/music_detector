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
