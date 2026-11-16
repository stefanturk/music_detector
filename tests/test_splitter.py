"""The shared demucs runner. demucs is faked by a tiny script, so these run without torch."""
import os
import shlex
import subprocess
import sys
import threading
import time

import pytest

from music_detector import splitter

FAKE = r'''
import os, sys, time
a = sys.argv[1:]
open(CALLS, "a").write(" ".join(a) + "\n")
out = a[a.index("-o") + 1]; model = a[a.index("-n") + 1]; song = a[-1]
mode = open(MODE).read().strip()
if mode == "slow":
    time.sleep(30)
if mode == "fail":
    print("boom: out of memory", flush=True); sys.exit(3)
if mode != "nostems":
    d = os.path.join(out, model, os.path.splitext(os.path.basename(song))[0])
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "drums.wav"), "w").write("x")
sys.stdout.write("  5%|#  | 1/20\r 45%|####| 9/20\r100%|####| 20/20\n"); sys.stdout.flush()
'''


@pytest.fixture
def fake(tmp_path):
    (tmp_path / "mode").write_text("ok")
    script = tmp_path / "fake_demucs.py"
    script.write_text(f"CALLS = {str(tmp_path / 'calls.txt')!r}\nMODE = {str(tmp_path / 'mode')!r}\n" + FAKE)
    py = tmp_path / "py"
    py.write_text(f"#!/bin/sh\nshift 2\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} \"$@\"\n")
    py.chmod(0o755)
    song = tmp_path / "Song One.mp3"
    song.write_text("x")
    out = tmp_path / "out"
    out.mkdir()

    class F:
        pass
    f = F()
    f.py, f.song, f.out = str(py), str(song), str(out)
    f.calls = lambda: (tmp_path / "calls.txt").read_text().splitlines()
    f.mode = lambda m: (tmp_path / "mode").write_text(m)
    return f


def test_returns_the_stem_folder_and_reports_each_percent(fake):
    got = []
    d = splitter.run_demucs(fake.py, fake.song, fake.out, "htdemucs", on_percent=got.append)
    assert d == os.path.join(fake.out, "htdemucs", "Song One")
    assert os.path.exists(os.path.join(d, "drums.wav"))
    assert got == [5, 45, 100]


def test_device_and_shifts_are_only_passed_when_asked_for(fake):
    splitter.run_demucs(fake.py, fake.song, fake.out, "htdemucs", on_percent=lambda p: None)
    plain = fake.calls()[-1]
    assert "-d" not in plain.split() and "--shifts" not in plain
    splitter.run_demucs(fake.py, fake.song, fake.out, "htdemucs_6s", device="cpu", shifts=0,
                        two_stems="vocals", on_percent=lambda p: None)
    c = fake.calls()[-1].split()
    assert c[c.index("-d") + 1] == "cpu" and c[c.index("--shifts") + 1] == "0"
    assert c[c.index("-n") + 1] == "htdemucs_6s" and c[c.index("--two-stems") + 1] == "vocals"


def test_a_failure_carries_demucs_last_words(fake):
    fake.mode("fail")
    with pytest.raises(subprocess.CalledProcessError) as e:
        splitter.run_demucs(fake.py, fake.song, fake.out, "htdemucs", on_percent=lambda p: None)
    assert "out of memory" in e.value.output


def test_exit_zero_but_no_stems_is_an_error(fake):
    fake.mode("nostems")
    with pytest.raises(RuntimeError, match="wrote no stems"):
        splitter.run_demucs(fake.py, fake.song, fake.out, "htdemucs", on_percent=lambda p: None)


def test_cancel_stops_demucs_quickly_and_raises(fake):
    fake.mode("slow")
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    t0 = time.time()
    with pytest.raises(splitter.Cancelled):
        splitter.run_demucs(fake.py, fake.song, fake.out, "htdemucs", should_cancel=flag.is_set)
    assert time.time() - t0 < 10


def test_cancelled_is_not_swallowed_by_except_exception():
    assert issubclass(splitter.Cancelled, BaseException) and not issubclass(splitter.Cancelled, Exception)


def test_module_cmd_can_wrap_the_launch(fake):
    seen = []

    def wrap(py, module):
        seen.append(module)
        return [py, "-m", module]
    splitter.run_demucs(fake.py, fake.song, fake.out, "htdemucs", on_percent=lambda p: None, module_cmd=wrap)
    assert seen == ["demucs"]


def test_speed_note_only_on_an_intel_mac():
    assert "processor alone" in splitter.speed_note("darwin", "x86_64")
    assert splitter.speed_note("darwin", "arm64") == ""
    assert splitter.speed_note("win32", "AMD64") == ""
    assert splitter.speed_note("linux", "x86_64") == ""
