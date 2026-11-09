"""The shared song renderer: right files, right lengths, fades, re-render only what changed."""
import json
import os
import subprocess

import pytest

from music_detector import songs as S

FF = S.find_ffmpeg()
pytestmark = pytest.mark.skipif(FF is None, reason="ffmpeg not installed")


def _tone(path, secs=30):
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={secs}", "-ac", "2", "-ar", "48000", str(path)],
                   check=True)


def _dur(path):
    r = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True)
    t = r.stderr.split("Duration: ")[1].split(",")[0]
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


SONGS = [{"number": 1, "name": "Black Heart", "start": 2.0, "end": 8.0},
         {"number": 2, "name": "Sure Dont Miss Me / Live", "start": 10.0, "end": 14.5}]


def test_export_writes_named_files_of_the_right_length(tmp_path):
    src = tmp_path / "show.wav"
    _tone(src)
    out = tmp_path / "Songs"
    res = S.export(str(src), SONGS, str(out))
    assert res["ok"], res
    names = sorted(os.listdir(out))
    assert names == [S.MARKER, "01 - Black Heart.mp3", "01 - Black Heart.wav",
                     "02 - Sure Dont Miss Me Live.mp3", "02 - Sure Dont Miss Me Live.wav"]
    assert abs(_dur(out / "01 - Black Heart.wav") - 6.0) < 0.05
    assert abs(_dur(out / "02 - Sure Dont Miss Me Live.wav") - 4.5) < 0.05
    assert S.is_export_dir(str(out))


def test_only_changed_songs_render_again_and_gone_ones_are_removed(tmp_path):
    src = tmp_path / "show.wav"
    _tone(src)
    out = tmp_path / "Songs"
    S.export(str(src), SONGS, str(out))
    res = S.export(str(src), SONGS, str(out))
    assert res["written"] == [] and len(res["kept"]) == 4
    moved = [dict(SONGS[0], end=9.0), dict(SONGS[1], name="Renamed")]
    res = S.export(str(src), moved, str(out))
    assert sorted(res["written"]) == ["01 - Black Heart.mp3", "01 - Black Heart.wav",
                                      "02 - Renamed.mp3", "02 - Renamed.wav"]
    assert sorted(res["removed"]) == ["02 - Sure Dont Miss Me Live.mp3",
                                      "02 - Sure Dont Miss Me Live.wav"]
    man = json.loads((out / S.MARKER).read_text())
    assert set(man["songs"]) == {"01 - Black Heart", "02 - Renamed"}


def test_a_song_without_a_window_is_reported_not_guessed(tmp_path):
    src = tmp_path / "show.wav"
    _tone(src, 5)
    res = S.export(str(src), [{"number": 1, "name": "X"}], str(tmp_path / "o"))
    assert not res["ok"] and "no start/end" in res["errors"][0]
