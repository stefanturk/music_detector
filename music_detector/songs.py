"""Cut a show's one long recording into its songs: one WAV (+ MP3) per song, with short fades.

Shared by Movie Maker (Song labeler -> "Export songs") and Ableton Loader (its standalone song
splitter), so both write the same files the same way:

    from music_detector import songs
    res = songs.export("/path/Full Show.wav",
                       [{"number": 1, "name": "Black Heart", "start": 812.4, "end": 1009.1}, ...],
                       "/path/Show Songs", progress=lambda i, n, name: ...)

Files are named "01 - Black Heart.wav" / ".mp3". The folder gets a MARKER manifest recording each
song's window, so a later export only re-renders songs whose window (or source) changed, removes
the files of songs that are gone, and lets apps recognise the folder as an export (not show media).
"""
import json
import os
import re
import shutil
import subprocess

MARKER = ".song_export.json"
FADE_S = 0.05
FORMATS = ("wav", "mp3")
MP3_BITRATE = "320k"


def find_ffmpeg():
    for p in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if p and os.path.exists(p):
            return p
    return None


def safe_name(s):
    """A file-name-safe song title (no slashes/colons; keeps letters, spaces, punctuation)."""
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", str(s or "")).strip().strip(".")
    return re.sub(r"\s+", " ", s) or "Song"


def stem_for(song, i):
    num = song.get("number")
    num = int(num) if isinstance(num, (int, float)) or str(num or "").isdigit() else i + 1
    return f"{num:02d} - {safe_name(song.get('name') or f'Song {num}')}"


def is_export_dir(path):
    return os.path.isfile(os.path.join(path, MARKER))


def read_manifest(out_dir):
    try:
        with open(os.path.join(out_dir, MARKER), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_manifest(out_dir, d):
    p = os.path.join(out_dir, MARKER)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, p)


def _render(ffmpeg, source, start, dur, fade, out_path, fmt):
    tmp = out_path + ".part." + fmt                  # (ffmpeg picks the format by extension)
    fade = max(0.0, min(fade, dur / 3.0))
    af = f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:.3f}"
    codec = (["-c:a", "pcm_s24le"] if fmt == "wav" else
             ["-c:a", "libmp3lame", "-b:a", MP3_BITRATE])
    cmd = [ffmpeg, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", source,
           "-af", af] + codec + [tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
        return (r.stderr or "ffmpeg failed").strip()[-300:]
    os.replace(tmp, out_path)
    return None


def export(source, songs, out_dir, *, fade_s=FADE_S, formats=FORMATS, ffmpeg=None, progress=None,
           force=False):
    """Render each song [start, end] (seconds into `source`) into `out_dir`. Returns
    {"ok", "dir", "written": [file...], "kept": [file...], "removed": [file...], "errors": [msg...]}.
    `progress(i, n, title)` is called before each song. Deterministic file names; songs with no
    window or a window past the recording's start are reported, not guessed."""
    res = {"ok": False, "dir": out_dir, "written": [], "kept": [], "removed": [], "errors": []}
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        res["errors"].append("ffmpeg isn't installed")
        return res
    if not (source and os.path.isfile(source)):
        res["errors"].append("the recording isn't there")
        return res
    os.makedirs(out_dir, exist_ok=True)
    man = read_manifest(out_dir)
    old = man.get("songs") or {}
    src_sig = [os.path.abspath(source), os.path.getsize(source)]
    new, todo = {}, []
    for i, s in enumerate(songs or []):
        try:
            a, b = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            res["errors"].append(f"{s.get('name') or f'song {i + 1}'}: no start/end")
            continue
        if b - a < 1.0 or a < 0:
            res["errors"].append(f"{s.get('name') or f'song {i + 1}'}: window {a:.1f}-{b:.1f} s")
            continue
        stem = stem_for(s, i)
        new[stem] = {"start": round(a, 3), "end": round(b, 3), "source": src_sig,
                     "formats": sorted(formats)}
        todo.append((stem, a, b, s.get("name") or stem))
    # files of songs that are gone (renamed / removed) — only ones this exporter wrote
    for stem, meta in old.items():
        if stem not in new:
            for fmt in meta.get("formats") or FORMATS:
                p = os.path.join(out_dir, f"{stem}.{fmt}")
                if os.path.exists(p):
                    os.remove(p)
                    res["removed"].append(os.path.basename(p))
    done = {k: v for k, v in old.items() if k in new}
    for i, (stem, a, b, title) in enumerate(todo):
        if progress:
            progress(i, len(todo), title)
        files = [os.path.join(out_dir, f"{stem}.{fmt}") for fmt in formats]
        same = (not force and old.get(stem) == new[stem] and all(os.path.exists(p) for p in files))
        if same:
            res["kept"] += [os.path.basename(p) for p in files]
            continue
        bad = None
        for fmt, p in zip(formats, files):
            bad = _render(ffmpeg, source, a, b - a, fade_s, p, fmt)
            if bad:
                res["errors"].append(f"{title}: {bad}")
                break
            res["written"].append(os.path.basename(p))
        if not bad:
            done[stem] = new[stem]
        _write_manifest(out_dir, {"songs": done, "source": src_sig[0]})
    _write_manifest(out_dir, {"songs": done, "source": src_sig[0]})
    res["ok"] = not res["errors"]
    return res
