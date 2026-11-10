"""The parts splitter: FLAC naming, padding, manifest skip / re-run. demucs is faked by a tiny
script that writes the six source WAVs, so these run without torch."""
import json
import os
import shlex
import subprocess
import sys

import pytest

from music_detector import separate as SEP

FF = SEP._ffmpeg()
pytestmark = pytest.mark.skipif(FF is None, reason="ffmpeg not installed")

FAKE = r'''
import os, subprocess, sys
a = sys.argv[1:]
out = a[a.index("-o") + 1]; model = a[a.index("-n") + 1]; wav = a[-1]
assert "--shifts" in a and a[a.index("--shifts") + 1] == "0"
d = os.path.join(out, model, os.path.splitext(os.path.basename(wav))[0])
os.makedirs(d, exist_ok=True)
srcs = ["vocals", "no_vocals"] if "--two-stems" in a else ["vocals", "drums", "bass", "guitar", "piano", "other"]
for s in srcs:
    subprocess.run([FF, "-y", "-v", "error", "-i", wav, os.path.join(d, s + ".wav")], check=True)
print("50%\r100%", flush=True)
with open(CALLS, "a") as f:
    f.write(model + "\n")
'''


@pytest.fixture
def fake_py(tmp_path, monkeypatch):
    """A 'python' that, when run as `-m demucs ...`, runs the fake demucs above."""
    script = tmp_path / "fake_demucs.py"
    script.write_text(f"FF = {FF!r}\nCALLS = {str(tmp_path / 'calls.txt')!r}\n" + FAKE)
    py = tmp_path / "py"
    py.write_text(f"#!/bin/sh\nshift 2\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} \"$@\"\n")
    py.chmod(0o755)
    return str(py)


def _tone(path, secs=30):
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=330:duration={secs}", "-ac", "2", "-ar", "44100", str(path)],
                   check=True)


def _dur(path):
    r = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True)
    h, m, s = r.stderr.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_split_writes_six_mono_flacs_with_padding(tmp_path, fake_py):
    src = tmp_path / "show.wav"
    _tone(src, 40)
    out = tmp_path / "parts" / "1 - Song"
    seen = []
    res = SEP.split_song(str(src), 5.0, 15.0, str(out), python=fake_py, progress=seen.append)
    assert res["ok"], res
    assert sorted(res["files"]) == sorted(f"{p}.flac" for p in
                                          ("vocals", "drums", "bass", "guitar", "keys", "other"))
    assert res["pad_s"] == SEP.PAD_BEFORE_S
    assert seen[-1] == 100
    f = out / "keys.flac"
    probe = subprocess.run([FF, "-i", str(f)], capture_output=True, text=True).stderr
    assert "22050 Hz, mono" in probe
    assert abs(_dur(f) - (10.0 + SEP.PAD_BEFORE_S + SEP.PAD_AFTER_S)) < 0.1


def test_song_near_the_start_gets_a_shorter_pad(tmp_path, fake_py):
    src = tmp_path / "show.wav"
    _tone(src, 20)
    res = SEP.split_song(str(src), 0.4, 6.0, str(tmp_path / "p"), python=fake_py)
    assert res["ok"] and res["pad_s"] == 0.4


def test_same_inputs_skip_and_changed_window_reruns(tmp_path, fake_py):
    src = tmp_path / "show.wav"
    _tone(src, 30)
    out = tmp_path / "p"
    assert SEP.split_song(str(src), 2.0, 8.0, str(out), python=fake_py)["ok"]
    again = SEP.split_song(str(src), 2.0, 8.0, str(out), python=fake_py)
    assert again["ok"] and again["kept"]
    calls = (tmp_path / "calls.txt").read_text().split()
    assert calls == [SEP.MODEL]
    inside = SEP.split_song(str(src), 2.5, 8.0, str(out), python=fake_py)   # inside: kept
    assert inside["ok"] and inside["kept"]
    moved = SEP.split_song(str(src), 0.5, 8.0, str(out), python=fake_py)    # earlier than split
    assert moved["ok"] and not moved["kept"]
    assert json.load(open(out / SEP.MARKER))["sig"]["window"] == [0.5, 8.0]


def test_missing_demucs_or_source_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(SEP, "find_python", lambda: None)
    src = tmp_path / "show.wav"
    _tone(src, 5)
    assert "isn't installed" in SEP.split_song(str(src), 0, 3, str(tmp_path / "p"))["error"]
    assert "isn't there" in SEP.split_song(str(tmp_path / "nope.wav"), 0, 3,
                                           str(tmp_path / "p"))["error"]


def test_failed_demucs_leaves_no_manifest(tmp_path):
    bad = tmp_path / "py"
    bad.write_text("#!/bin/sh\necho boom; exit 3\n")
    bad.chmod(0o755)
    src = tmp_path / "show.wav"
    _tone(src, 5)
    out = tmp_path / "p"
    res = SEP.split_song(str(src), 0, 3, str(out), python=str(bad))
    assert not res["ok"] and "boom" in res["error"]
    assert not (out / SEP.MARKER).exists()


def test_clean_vocal_returns_same_length(tmp_path, fake_py):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    x = np.sin(np.linspace(0, 400, 16000 * 3)).astype("float32") * 0.5
    v = SEP.clean_vocal(x, 16000, python=fake_py)
    assert v is not None and len(v) == len(x)


def test_analyze_parts_reads_the_song_window(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from music_detector.core import whoplays as W
    r = 22050
    t = np.arange(int(r * 12)) / r
    loud = (t >= 3.0) & (t < 8.0)                      # vocal sings 2-7 s of the song (pad 1 s)
    sf.write(tmp_path / "vocals.flac", (np.sin(2 * np.pi * 220 * t) * loud * .5).astype("float32"), r)
    sf.write(tmp_path / "drums.flac", (np.sin(2 * np.pi * 60 * t) * .3).astype("float32"), r)
    tl = W.analyze_parts(str(tmp_path), 1.0, 10.0, str(tmp_path / "cache"),
                         voices={"VOCAL": ["vocals.flac"], "DRUMS": ["drums.flac"]},
                         song="1 - Song", truenorth_offset_s=100.0)
    assert tl["stem_offset_s"] == 1.0 and tl["song_len_s"] == 10.0
    assert tl["truenorth_offset_s"] == 100.0
    assert any(sp["lead"] == "VOCAL" for sp in tl["spans"])


def test_a_window_moved_inside_the_split_keeps_the_parts(tmp_path, fake_py):
    src = tmp_path / "show.wav"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:d=60", str(src)], check=True)
    out = tmp_path / "parts"
    r = SEP.split_song(str(src), 10.0, 40.0, str(out), python=fake_py)
    assert r["ok"] and not r["kept"]
    calls = (tmp_path / "calls.txt").read_text().count("\n")
    r2 = SEP.split_song(str(src), 15.0, 38.0, str(out), python=fake_py)    # trimmed both ends
    assert r2["ok"] and r2["kept"] and r2["pad_s"] == 6.0
    assert (tmp_path / "calls.txt").read_text().count("\n") == calls       # no new split
    assert SEP.is_current(str(src), 15.0, 38.0, str(out))
    r3 = SEP.split_song(str(src), 5.0, 38.0, str(out), python=fake_py)      # starts earlier: re-split
    assert r3["ok"] and not r3["kept"]
