"""EQ ownership on synthetic mics: a mic owns its range only when its own source plays."""
import numpy as np

from music_detector.core import eq

R = eq.RATE


def test_a_mic_owns_its_range_only_while_its_source_plays():
    t = np.arange(R * 20) / R
    band = 0.5 * np.sin(2 * np.pi * 120 * t)                 # the band: low, loud, always on
    voice = 0.3 * np.sin(2 * np.pi * 3000 * t) * ((t // 4) % 2 == 1)   # sings every other 4 s
    mic = voice + 0.05 * band                                # her mic hears the band too
    own = eq.owned(eq.band_bins(mic, 2000, 5000), eq.band_bins(band + 0.02 * voice, 2000, 5000) + 1e-6)
    sung = ((np.arange(len(own)) * eq.BIN_S // 4) % 2 == 1)
    assert (own == sung).mean() > 0.9


def test_cant_judge_without_audio():
    assert eq.voice_ownership({"VOCAL": ["a.wav"]}, "/nonexistent", 0, 10, 200, 0.05) is None
    assert eq.voice_ownership({"GUITAR": ["g.wav"]}, "/x", 0, 10, 200, 0.05) is None


def _wav(path, x):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(R)
        w.writeframes((np.clip(x, -1, 1) * 32000).astype("<i2").tobytes())


def test_the_bleed_guard_silences_a_horn_mic_that_only_hears_the_band(tmp_path):
    import shutil
    if not shutil.which("ffmpeg") and not __import__("os").path.exists("/opt/homebrew/bin/ffmpeg"):
        return
    t = np.arange(R * 20) / R
    on = lambda a, b: ((t >= a) & (t < b)).astype(float)
    trumpet = 0.4 * np.sin(2 * np.pi * 2000 * t) * (on(4, 8) + on(12, 16))
    sax = 0.4 * np.sin(2 * np.pi * 1500 * t) * on(8, 12)
    noise = lambda seed: 0.003 * np.random.RandomState(seed).randn(len(t))   # every real mic has a floor
    _wav(tmp_path / "trumpet.wav", trumpet + 0.05 * sax + noise(1))
    _wav(tmp_path / "sax.wav", sax + 0.05 * trumpet + noise(2))     # the sax mic hears the trumpet
    _wav(tmp_path / "gtr.wav", 0.2 * np.sin(2 * np.pi * 200 * t) + noise(3))
    voices = {"SAX": ["sax.wav"], "TRUMPET": ["trumpet.wav"], "GUITAR": ["gtr.wav"]}
    n, hop = 400, 0.05
    loud = lambda a, b: np.array([1.0 if a <= i * hop < b else 0.0 for i in range(n)])
    act = {"SAX": np.maximum(loud(4, 8), np.maximum(loud(8, 12), loud(12, 16))) * 0.9,   # bleed counts
           "TRUMPET": loud(4, 8) + loud(12, 16), "GUITAR": np.ones(n)}
    out = eq.guard_activity(act, voices, str(tmp_path), 0, n, hop, ffmpeg=None)
    sax_on = out["SAX"] > 0
    assert sax_on[int(9 / hop):int(11 / hop)].all()                # really playing: kept
    assert not sax_on[int(5 / hop):int(7 / hop)].any()             # trumpet bleed: gone
    assert not sax_on[int(13 / hop):int(15 / hop)].any()
    assert (out["GUITAR"] == act["GUITAR"]).all()                  # other voices untouched
    assert (act["SAX"][int(5 / hop):int(7 / hop)] > 0).all()       # the input wasn't changed
