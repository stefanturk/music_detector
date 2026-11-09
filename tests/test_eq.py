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
