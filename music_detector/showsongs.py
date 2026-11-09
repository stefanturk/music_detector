"""Which songs a Movie Maker show has marked in a recording, read-only.

Ableton Loader's song splitter calls find(recording) before it hunts for songs itself: when the
recording is a Movie Maker show's Northstar Recording, the songs come back already named and timed
(the same windows Movie Maker's own "Export songs" uses), plus the folder both apps export into.

A Movie Maker show keeps its data in "<Show> Data/project.json" inside the show folder. Its paths
are stored show-relative, "{show}/..." (the show folder = the Data folder's parent). Nothing here
writes anything.
"""
import glob
import json
import os

DATA_SUFFIX = " Data"
PROJECT_FILE = "project.json"


def _real(p):
    try:
        return os.path.realpath(p)
    except (OSError, ValueError):
        return p


def _resolve(path, show_root):
    path = str(path or "").strip()
    if not path:
        return ""
    if "{show}" in path:
        path = path.replace("{show}", show_root)
    elif not os.path.isabs(path):
        path = os.path.join(show_root, path)
    return path


def _projects_near(recording):
    """project.json files of shows in the recording's folder or the folder above it (the
    recording may sit in the show folder or in one of its sub-folders)."""
    here = os.path.dirname(os.path.abspath(recording))
    seen = []
    for d in (here, os.path.dirname(here)):
        for data in sorted(glob.glob(os.path.join(glob.escape(d), "*" + DATA_SUFFIX))):
            pf = os.path.join(data, PROJECT_FILE)
            if os.path.isfile(pf) and pf not in seen:
                seen.append(pf)
    return seen


def _window(s):
    win = s.get("window")
    if isinstance(win, (list, tuple)) and len(win) == 2 and None not in win:
        return float(win[0]), float(win[1])
    a, b = s.get("start_fullshow_s"), s.get("end_fullshow_s")
    if a is not None and b is not None:
        return float(a), float(b)
    return None


def _used(s):
    return s.get("used", True) is not False and not s.get("skip")


def find(recording):
    """{"show", "band", "event", "folder", "songs": [{number, name, start, end, use}]} when
    `recording` is a Movie Maker show's Northstar Recording with songs marked, else None.
    Times are seconds into the recording; `folder` is the show's "<Show> Songs" export folder."""
    if not (recording and os.path.isfile(recording)):
        return None
    want = _real(recording)
    for pf in _projects_near(recording):
        try:
            with open(pf, encoding="utf-8") as f:
                proj = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(proj, dict):
            continue
        root = os.path.dirname(os.path.dirname(pf))
        paths = proj.get("paths") or {}
        mine = [_resolve(paths.get(k), root) for k in ("northstar_audio", "fullshow_audio")]
        if not any(p and _real(p) == want for p in mine):
            continue
        songs = []
        order = sorted(proj.get("songs") or [], key=lambda s: (s.get("number") is None,
                                                              s.get("number") or 0))
        for i, s in enumerate(order):
            w = _window(s)
            if not w or w[1] - w[0] < 1.0:
                continue
            songs.append({"number": s.get("number") or i + 1,
                          "name": (s.get("name") or "").strip() or f"Song {i + 1}",
                          "start": round(max(0.0, w[0]), 2), "end": round(w[1], 2),
                          "use": _used(s)})
        if not songs:
            return None
        name = proj.get("name") or proj.get("slug") or os.path.basename(root)
        out = _resolve(paths.get("default_output_dir") or "{show}", root)
        return {"show": name, "band": proj.get("band") or None, "event": proj.get("event") or None,
                "folder": os.path.join(out, f"{name} Songs"), "songs": songs}
    return None
