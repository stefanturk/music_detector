"""The one place that runs demucs (the song splitter) for every Steezy app.

Beat Thief and Movie Maker (through `separate`) both call this, so there is one copy of the hard
parts: starting demucs, reading its progress, stopping it on cancel, finding a Python that has it,
and the "this Mac is slow at this" hint. What each app does with the stems afterwards (Beat
Thief's WAV folders, Movie Maker's FLAC parts) stays in that app.

Standard library only, so any app can import it without pulling in numpy or torch. demucs itself
runs in a separate Python (a 2 GB install), found by `find_python()` or passed in.

    from music_detector import splitter
    stems = splitter.run_demucs(sys.executable, "song.mp3", out_dir, "htdemucs")   # -> folder of <source>.wav
"""
import importlib.util
import os
import platform
import re
import subprocess
import sys
import threading

MODEL_4 = "htdemucs"           # vocals, drums, bass, other
MODEL_6 = "htdemucs_6s"        # vocals, drums, bass, guitar, piano, other

_PCT = re.compile(r"(\d{1,3})%")      # demucs' tqdm frames look like "  45%|####   | 12/27 [...]"
_CANCEL_POLL_S = 0.2


class Cancelled(BaseException):
    """A run was stopped on request. A BaseException (like KeyboardInterrupt) on purpose: callers
    wrap their work in broad `except Exception` handlers that turn failures into a warning and
    carry on to the next song, and a cancel must cut straight through those."""


def _candidates():
    home = os.path.expanduser("~")
    return [sys.executable, "/usr/bin/python3", "/opt/homebrew/bin/python3", "/usr/local/bin/python3",
            os.path.join(home, "Library/Application Support/Steezy Studio/engine/python/bin/python3")]


_found = {}


def find_python():
    """The first Python on this machine that can run demucs, else None (cached for the session)."""
    if "py" in _found:
        return _found["py"]
    py = None
    seen = set()
    for p in _candidates():
        if not p or p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        try:
            r = subprocess.run([p, "-c", "import demucs"], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            py = p
            break
    _found["py"] = py
    return py


def available():
    return find_python() is not None


def installed_here():
    """Whether THIS Python has torch and demucs. Looked for, not imported (importing torch takes
    seconds and finding it is all this needs)."""
    return all(importlib.util.find_spec(n) is not None for n in ("torch", "demucs"))


def speed_note(system=None, machine=None):
    """What to tell someone on an Intel Mac (no graphics path for torch there, so it runs on the
    processor alone and takes several times as long as on an M-series Mac). Empty elsewhere. An
    Intel-built Python on an M-series Mac reports x86_64 too and is just as slow, so the same words
    fit. The "5-10 minutes" is an estimate, not a measurement."""
    if (system or sys.platform) == "darwin" and (machine or platform.machine()) == "x86_64":
        return ("This Mac splits songs on its processor alone, so expect roughly 5-10 minutes "
                "per song, and closing other apps helps.")
    return ""


def _plain_cmd(py, module):
    return [py, "-m", module]


def run_demucs(py, input_path, out_dir, model, *, device=None, shifts=None, two_stems=None,
               on_percent=None, should_cancel=None, module_cmd=None):
    """Run demucs on `input_path`; returns the folder holding its <source>.wav files.

    device    "cpu"/"cuda"/... or None to let demucs choose (CUDA if there is one, else the CPU; it
              never picks the Mac GPU, so on a Mac both come to the same thing).
    shifts    None for demucs' own default; 0 makes it deterministic (no random time shift).
    on_percent(pct)   gets each new 0-100 as demucs reports it. Without it, a one-line bar is drawn
              on stdout.
    should_cancel()   polled on its own thread; True kills demucs and raises Cancelled. A thread,
              because demucs' first minute (importing torch, loading the model) prints nothing and
              that is exactly when people change their mind.
    module_cmd(py, "demucs") -> argv  lets a caller wrap the launch (Beat Thief hides console
              windows on Windows this way). Default is plain `py -m demucs`."""
    cmd = list((module_cmd or _plain_cmd)(py, "demucs")) + ["-n", model, "-o", out_dir]
    if device:
        cmd += ["-d", device]
    if shifts is not None:
        cmd += ["--shifts", str(int(shifts))]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(input_path)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    killed = threading.Event()
    finished = threading.Event()
    watcher = None
    if should_cancel is not None:
        def watch():
            while not finished.wait(_CANCEL_POLL_S):
                if should_cancel():
                    killed.set()
                    proc.kill()
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()

    buf, last, tail = "", None, []
    # read1, not read: read() waits for 4096 bytes, which is dozens of progress frames, so the bar
    # would move in jumps. Whole \r/\n-terminated frames only, so "NN%" is never cut in half.
    for chunk in iter(lambda: proc.stdout.read1(4096), b""):
        if killed.is_set():
            break
        text = chunk.decode(errors="ignore")
        buf += text
        tail = (tail + [text])[-20:]
        frames = re.split(r"[\r\n]", buf)
        buf = frames.pop()
        for frame in frames:
            found = _PCT.findall(frame)
            if not found:
                continue
            pct = min(int(found[-1]), 100)
            if pct == last:
                continue
            last = pct
            if on_percent is not None:
                on_percent(pct)
            else:
                filled = int(30 * pct / 100)
                sys.stdout.write(f"\r  [{'#' * filled}{'-' * (30 - filled)}] {pct:3d}%")
                sys.stdout.flush()
    proc.stdout.close()
    proc.wait()
    finished.set()
    if watcher is not None:
        watcher.join(timeout=1)
    if killed.is_set():
        raise Cancelled()
    if last is not None and on_percent is None:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=("".join(tail).strip() or None))

    track = os.path.splitext(os.path.basename(input_path))[0]
    stem_dir = os.path.join(out_dir, model, track)
    # demucs can exit 0 and still not write where we expect (a name it changed, a model that lays
    # its output out differently). Saying so beats a bare "no such file" later.
    if not os.path.isdir(stem_dir):
        raise RuntimeError(f"demucs finished but wrote no stems for {track!r} (expected them in {stem_dir})")
    return stem_dir
