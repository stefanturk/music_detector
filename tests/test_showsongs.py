"""showsongs.find: a Movie Maker show's songs for its Northstar Recording, read-only."""
import json
import os

from music_detector import showsongs


def _show(tmp_path, songs, ns="{show}/Gig Full Show.mp3"):
    root = tmp_path / "Gig"
    (root / "Gig Data").mkdir(parents=True)
    rec = root / "Gig Full Show.mp3"
    rec.write_bytes(b"x")
    proj = {"name": "Gig", "slug": "gig", "band": "Litty deBungus",
            "paths": {"northstar_audio": ns, "default_output_dir": "{show}"}, "songs": songs}
    pf = root / "Gig Data" / "project.json"
    pf.write_text(json.dumps(proj))
    return root, rec, pf


def test_a_shows_northstar_recording_brings_its_songs(tmp_path):
    songs = [{"number": 2, "name": "Bad Bad News", "window": [200.0, 380.5], "used": True},
             {"number": 1, "name": "Black Heart", "start_fullshow_s": 0.4, "end_fullshow_s": 195.1},
             {"number": 3, "name": "Banter", "window": [381.0, 400.0], "used": False}]
    root, rec, pf = _show(tmp_path, songs)
    before = pf.read_text()
    r = showsongs.find(str(rec))
    assert r["show"] == "Gig" and r["band"] == "Litty deBungus"
    assert r["folder"] == os.path.join(str(root), "Gig Songs")
    assert [(s["number"], s["name"], s["start"], s["end"], s["use"]) for s in r["songs"]] == [
        (1, "Black Heart", 0.4, 195.1, True), (2, "Bad Bad News", 200.0, 380.5, True),
        (3, "Banter", 381.0, 400.0, False)]
    assert pf.read_text() == before                       # never writes


def test_a_recording_in_a_sub_folder_is_found_too(tmp_path):
    root = tmp_path / "Gig"
    (root / "Gig Data").mkdir(parents=True)
    (root / "Audio").mkdir()
    rec = root / "Audio" / "mix.wav"
    rec.write_bytes(b"x")
    (root / "Gig Data" / "project.json").write_text(json.dumps(
        {"name": "Gig", "paths": {"northstar_audio": "{show}/Audio/mix.wav"},
         "songs": [{"number": 1, "name": "A", "window": [1.0, 60.0]}]}))
    assert showsongs.find(str(rec))["songs"][0]["name"] == "A"


def test_other_recordings_and_shows_without_songs_give_none(tmp_path):
    root, rec, pf = _show(tmp_path, [{"number": 1, "name": "A", "window": [1.0, 60.0]}])
    other = root / "Other.wav"
    other.write_bytes(b"x")
    assert showsongs.find(str(other)) is None
    _show(tmp_path / "b", [])
    assert showsongs.find(str(tmp_path / "b" / "Gig" / "Gig Full Show.mp3")) is None
    assert showsongs.find(str(tmp_path / "nowhere.wav")) is None
