"""The shared song finder (moved from movie_maker's set_splitter) still finds its synthetic set."""
from music_detector import split


def test_selftest_recovers_every_song():
    assert split._selftest() == 0


import numpy as np


def _set(parts, hop=0.1):
    """A loudness curve from (seconds, dB) pieces, with a little wobble."""
    out = []
    for i, (sec, db) in enumerate(parts):
        n = int(round(sec / hop))
        out.append(db + 1.5 * np.sin(np.arange(n) * (0.7 + 0.1 * i)))
    return 10.0 ** (np.concatenate(out) / 20.0)


def test_a_stop_inside_a_song_does_not_split_it():
    # song 1 has a 3 s dead stop in the middle; a real 20 s gap before song 2
    rms = _set([(80, -10), (3, -60), (80, -10), (20, -45), (90, -10)])
    wins, _ = split.detect_windows(rms, 0.1)
    assert len(wins) == 2
    assert wins[0]["dur_s"] > 150


def test_talking_between_songs_still_counts_as_quiet():
    # a 14 s gap with someone talking into a mic (way under the band) still ends the song,
    # and the talking isn't found as a song of its own
    talk = [(1.0, -25), (0.8, -50)] * 8
    rms = _set([(90, -10)] + talk + [(90, -10)] + [(1.2, -25), (0.6, -50)] * 20 + [(90, -10)])
    wins, _ = split.detect_windows(rms, 0.1)
    assert len(wins) == 3
