"""Split a song out of a full mix into its parts (vocals, drums, bass, guitar, keys, other) with
demucs, for shows that have no per-instrument multitrack recordings.

Shared like `songs` / `split`: Movie Maker calls it per song; anything else can too.

    from music_detector import separate
    res = separate.split_song("/path/Full Show.mp3", 812.4, 1009.1, "/path/parts/1 - Black Heart")
    # res["ok"], res["dir"], res["files"], res["pad_s"]

The parts are written as mono 22.05 kHz FLAC (`vocals.flac`, `drums.flac`, `bass.flac`,
`guitar.flac`, `keys.flac`, `other.flac`) -- about 3 MB per part per song, instead of ~35 MB of
stereo WAV. Each part starts PAD_BEFORE_S before the song (so part time 0 = song start - pad_s) and
runs PAD_AFTER_S past its end (room for the ring-out detector). A `.separate.json` manifest records
the source, window and model, so a second call with the same inputs does nothing.

demucs is machine learning, used like Whisper for lyrics: run once, saved, and every build from the
saved parts comes out the same. `--shifts 0` keeps demucs itself deterministic too (its default
random time shift makes two runs differ slightly).

demucs runs in whichever Python has it (it's a 2 GB install: torch). `find_python()` looks.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

MARKER = ".separate.json"
MODEL = "htdemucs_6s"          # vocals, drums, bass, guitar, piano, other
CLEAN_MODEL = "htdemucs"       # the 4-source model; only its vocals are used by clean_vocal
RATE = 22050
PAD_BEFORE_S = 1.0
PAD_AFTER_S = 9.0              # the ending detector looks up to 8 s past the song label
PARTS = {"vocals": "vocals", "drums": "drums", "bass": "bass", "guitar": "guitar",
         "piano": "keys", "other": "other"}     # demucs source -> our file name
VERSION = 1

_PCT = re.compile(r"(\d{1,3})%")


class Cancelled(Exception):
    pass


def _candidates():
    home = os.path.expanduser("~")
    return [sys.executable, "/usr/bin/python3", "/opt/homebrew/bin/python3", "/usr/local/bin/python3",
            os.path.join(home, "Library/Application Support/Steezy Studio/engine/python/bin/python3")]


_found = {}


def find_python():
    """The first Python on this Mac that can run demucs, else None (cached for the session)."""
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


def _ffmpeg():
    for p in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if p and os.path.exists(p):
            return p
    return None


def read_manifest(out_dir):
    try:
        with open(os.path.join(out_dir, MARKER), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _sig(source, start, end, model):
    st = os.stat(source)
    return {"version": VERSION, "source": [os.path.abspath(source), st.st_size, int(st.st_mtime)],
            "window": [round(float(start), 3), round(float(end), 3)], "model": model}


def is_current(source, start, end, out_dir, model=MODEL):
    """True when out_dir already holds this song's parts from this source, window and model."""
    man = read_manifest(out_dir)
    try:
        sig = _sig(source, start, end, model)
    except OSError:
        return False
    return (man.get("sig") == sig and
            all(os.path.exists(os.path.join(out_dir, f)) for f in man.get("files") or ["-"]))


ENDING_ROOM_S = 8.0     # what the ending detector needs past the song's end


def covers(man):
    """[a, b]: the stretch of the recording (seconds) the saved parts hold, or None."""
    c = man.get("covers")
    if isinstance(c, list) and len(c) == 2:
        return float(c[0]), float(c[1])
    w = (man.get("sig") or {}).get("window")
    if isinstance(w, list) and len(w) == 2:
        return float(w[0]) - float(man.get("pad_s", 0.0)), float(w[1]) + PAD_AFTER_S
    return None


def reuse_inside(source, start, end, out_dir, model=MODEL):
    """A song whose start/end moved INSIDE what was already split keeps its parts: the manifest is
    re-pointed at the new window (pad_s = how far into the files the song now starts), nothing is
    re-split. Returns the updated manifest, or None when the parts don't cover the new window."""
    man = read_manifest(out_dir)
    if not man.get("files") or not all(os.path.exists(os.path.join(out_dir, f)) for f in man["files"]):
        return None
    try:
        sig = _sig(source, start, end, model)
    except OSError:
        return None
    old = man.get("sig") or {}
    if old.get("source") != sig["source"] or old.get("model") != model \
            or old.get("version") != VERSION:
        return None
    cv = covers(man)
    if cv is None:
        return None
    a, b = cv
    if float(start) < a - 1e-3 or float(end) + ENDING_ROOM_S > b + 1e-3 or float(end) <= float(start):
        return None
    man = dict(man, sig=sig, pad_s=round(float(start) - a, 3), covers=[round(a, 3), round(b, 3)])
    mp = os.path.join(out_dir, MARKER)
    with open(mp + ".tmp", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    os.replace(mp + ".tmp", mp)
    return man


def run_demucs(py, wav, out_dir, model, *, two_stems=None, progress=None, cancel=None):
    """Run demucs on `wav`; returns the folder holding its <source>.wav files."""
    cmd = [py, "-m", "demucs", "-n", model, "-d", "cpu", "--shifts", "0", "-o", out_dir]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(wav)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    killed = threading.Event()
    done = threading.Event()
    if cancel is not None:
        def watch():
            while not done.wait(0.2):
                if cancel():
                    killed.set()
                    proc.kill()
                    return
        threading.Thread(target=watch, daemon=True).start()
    buf, last, tail = "", None, []
    for chunk in iter(lambda: proc.stdout.read1(4096), b""):
        text = chunk.decode(errors="ignore")
        buf += text
        tail = (tail + [text])[-20:]
        frames = re.split(r"[\r\n]", buf)
        buf = frames.pop()
        for fr in frames:
            m = _PCT.findall(fr)
            if m:
                pct = min(int(m[-1]), 100)
                if pct != last:
                    last = pct
                    if progress:
                        progress(pct)
    proc.stdout.close()
    proc.wait()
    done.set()
    if killed.is_set():
        raise Cancelled()
    if proc.returncode != 0:
        raise RuntimeError(("".join(tail).strip() or "demucs failed")[-300:])
    d = os.path.join(out_dir, model, os.path.splitext(os.path.basename(wav))[0])
    if not os.path.isdir(d):
        raise RuntimeError(f"demucs wrote nothing for {os.path.basename(wav)}")
    return d


def _to_flac(ffmpeg, src, dst):
    tmp = dst + ".part.flac"
    r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", str(RATE),
                        "-sample_fmt", "s16", tmp], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError((r.stderr or "ffmpeg failed").strip()[-300:])
    os.replace(tmp, dst)


def split_song(source, start, end, out_dir, *, model=MODEL, progress=None, cancel=None,
               python=None, ffmpeg=None, force=False):
    """Split [start, end] (seconds into `source`) into its parts in `out_dir`.

    Returns {"ok", "dir", "files", "pad_s", "kept", "error"}. `pad_s` is how far before the song
    start the parts begin (less than PAD_BEFORE_S when the song starts near 0). `progress(pct)`
    gets demucs' 0-100; `cancel()` returning True stops it (raises Cancelled)."""
    res = {"ok": False, "dir": out_dir, "files": [], "pad_s": 0.0, "kept": False, "error": None}
    if not (source and os.path.isfile(source)):
        res["error"] = "the recording isn't there"
        return res
    if not force and is_current(source, start, end, out_dir, model):
        man = read_manifest(out_dir)
        res.update(ok=True, files=man["files"], pad_s=man.get("pad_s", 0.0), kept=True)
        return res
    if not force:
        kept = reuse_inside(source, start, end, out_dir, model)
        if kept is not None:
            res.update(ok=True, files=kept["files"], pad_s=kept["pad_s"], kept=True)
            return res
    py = python or find_python()
    if not py:
        res["error"] = "the splitter (demucs) isn't installed"
        return res
    ffmpeg = ffmpeg or _ffmpeg()
    if not ffmpeg:
        res["error"] = "ffmpeg isn't installed"
        return res
    a = max(0.0, float(start) - PAD_BEFORE_S)
    b = float(end) + PAD_AFTER_S
    pad = round(float(start) - a, 3)
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="separate_")
    try:
        wav = os.path.join(tmp, "song.wav")
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
                            "-i", source, "-ac", "2", "-ar", "44100", wav],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(wav):
            res["error"] = (r.stderr or "ffmpeg failed").strip()[-300:]
            return res
        got = run_demucs(py, wav, tmp, model, progress=progress, cancel=cancel)
        files = []
        for srcname, ours in PARTS.items():
            p = os.path.join(got, f"{srcname}.wav")
            if os.path.exists(p):
                _to_flac(ffmpeg, p, os.path.join(out_dir, f"{ours}.flac"))
                files.append(f"{ours}.flac")
        if not files:
            res["error"] = "the splitter wrote no parts"
            return res
        man = {"sig": _sig(source, start, end, model), "files": files, "pad_s": pad,
               "rate": RATE, "covers": [round(a, 3), round(b, 3)]}
        mp = os.path.join(out_dir, MARKER)
        with open(mp + ".tmp", "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        os.replace(mp + ".tmp", mp)
        res.update(ok=True, files=files, pad_s=pad)
        return res
    except Cancelled:
        raise
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        res["error"] = str(e)[-300:]
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def clean_vocal(mono, rate, *, python=None, cancel=None):
    """The vocal of a mono float track (numpy), with drum/horn bleed taken out by demucs. Returns a
    float32 array at `rate`, or None when demucs isn't available or fails (callers use the input)."""
    import numpy as np
    import soundfile as sf
    py = python or find_python()
    ffmpeg = _ffmpeg()
    if not py or not ffmpeg or mono is None or not len(mono):
        return None
    tmp = tempfile.mkdtemp(prefix="clean_vocal_")
    try:
        src = os.path.join(tmp, "mix.wav")
        sf.write(src, np.asarray(mono, dtype=np.float32), int(rate))
        got = run_demucs(py, src, tmp, CLEAN_MODEL, two_stems="vocals", cancel=cancel)
        out = os.path.join(tmp, "vocal.wav")
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", os.path.join(got, "vocals.wav"),
                            "-ac", "1", "-ar", str(int(rate)), "-c:a", "pcm_f32le", out],
                           capture_output=True)
        if r.returncode != 0:
            return None
        v, _ = sf.read(out, dtype="float32")
        n = len(mono)
        v = v[:n] if len(v) >= n else np.pad(v, (0, n - len(v)))
        return v.astype(np.float32)
    except Cancelled:
        raise
    except Exception:                           # noqa: BLE001
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
