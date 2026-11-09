"""The shared song finder (moved from movie_maker's set_splitter) still finds its synthetic set."""
from music_detector import split


def test_selftest_recovers_every_song():
    assert split._selftest() == 0
