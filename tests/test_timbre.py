"""Sound-type measurements on synthetic sounds whose answer we know."""
import numpy as np

from music_detector.core import timbre as T

R = T.RATE
t = np.arange(int(R * 6)) / R


def _norm(x):
    return x / np.max(np.abs(x)) * 0.5


def horn(vib_cents=25.0):
    """A blown note: rich harmonics, swells in and holds, with vibrato."""
    f = 330 * 2 ** (vib_cents / 1200 * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / R
    x = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 12))
    return _norm(x * np.clip(np.sin(np.pi * (t % 1.5) / 1.5) * 3, 0, 1))


def pluck():
    e = np.zeros_like(t)
    for s in np.arange(0, 6, 0.5):
        m = t >= s
        e[m] += np.exp(-(t[m] - s) * 6)
    return _norm(sum((0.6 / k) * np.sin(2 * np.pi * 220 * k * t) for k in range(1, 8)) * e)


def chord():
    return _norm(sum(sum((0.3 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6))
                     for f0 in (261.6, 329.6, 392.0)))


def med(m, k):
    return float(np.median(m[k]))


def test_pitch_is_found():
    f0, clar = T.yin(T._frames(_norm(np.sin(2 * np.pi * 440 * t))))
    assert abs(np.nanmedian(f0) - 440) < 2 and np.median(clar) > 0.9


def test_blown_notes_hold_plucked_notes_die():
    assert med(T.measure(horn()), "sustain") > 0.9
    assert med(T.measure(pluck()), "sustain") < 0.4


def test_vibrato_only_when_the_pitch_wobbles():
    assert med(T.measure(horn(25)), "vibrato") > 0.8
    assert med(T.measure(horn(0)), "vibrato") < 0.2


def test_a_chord_has_no_single_clear_pitch():
    assert med(T.measure(chord()), "clarity") < med(T.measure(horn()), "clarity") - 0.3


def test_a_horn_scores_above_a_pluck_and_a_chord():
    h = {k: np.median(T.horn_likeness(T.measure(x)))
         for k, x in (("horn", horn()), ("pluck", pluck()), ("chord", chord()))}
    assert h["horn"] > h["pluck"] + 0.3 and h["horn"] > h["chord"] + 0.3


def test_rumble_doesnt_hide_the_horn():
    """A close horn mic full of kick/bass bleed below 150 Hz still shows the horn's pitch."""
    rumble = 4.0 * np.sin(2 * np.pi * 55 * t)
    m = T.measure(horn() + rumble)
    assert med(m, "clarity") > 0.8


def test_same_answer_every_time_and_silence_is_quiet():
    a, b = T.measure(horn()), T.measure(horn())
    assert all(np.array_equal(a[k], b[k]) for k in a)
    q = T.measure(np.zeros(R * 2))
    assert not q["playing"].any() and not T.horn_likeness(q).any()
