"""music_detector.split — find the songs in one continuous live-set recording. No AI.

(Moved here from movie_maker's standalone set_splitter.py, so Movie Maker and Ableton Loader find
songs with ONE detector. It still imports only numpy + the standard library; ffmpeg is shelled out
to. movie_maker/set_splitter.py re-exports it, so old imports and its command line still work.)

Idea: a live set is loud, dense music separated by quieter gaps (applause,
banter, tuning). Take a smoothed loudness envelope and segment it with
hysteresis: enter a "song" when loudness rises above a high threshold, leave it
only after loudness stays below a low threshold for at least `min_gap_s`. The
hysteresis keeps quiet passages *inside* a song from splitting it, while a real
inter-song gap (sustained low energy) does split. Thresholds come from the
envelope's own percentiles, so it adapts to each show's level — no hard-coded dB.

It also CUTS the file into one piece per song (`cut`, stream copy, any container). For finished
song files with fades (WAV + MP3, named by the set list), use music_detector.songs.export.

    from music_detector import split
    windows, thr = split.detect_windows(rms, hop_s)      # pure, bring your own envelope
    windows, thr, meta = split.split("set.mov")          # decode + detect in one call
    paths = split.cut("set.mov", windows, "out_dir")     # export one file per song

    python3 -m music_detector.split selftest
    python3 -m music_detector.split <audio-or-video> [--json out.json] [--labels out.txt]
                                                     [--cut [out_dir]] [--reencode]
"""

import os
import subprocess

import numpy as np

# Loudness-over-time needs no pitch detail, so decode cheap and coarse.
DEFAULT_RATE = 2000          # Hz mono
DEFAULT_FRAME_MS = 100.0     # RMS frame = hop


# ----------------------------------------------------------------------------------
# Pure DSP — numpy only, no I/O. This is the reusable core.
# ----------------------------------------------------------------------------------
def rms_envelope(samples, rate, frame_ms=DEFAULT_FRAME_MS, hop_ms=None):
    """Framed RMS envelope. Returns (times_s, rms) at hop spacing."""
    if hop_ms is None:
        hop_ms = frame_ms
    frame = max(1, int(round(rate * frame_ms / 1000.0)))
    hop = max(1, int(round(rate * hop_ms / 1000.0)))
    n = len(samples)
    if n < frame:
        val = float(np.sqrt(np.mean(samples ** 2))) if n else 0.0
        return np.array([0.0]), np.array([val])
    starts = np.arange(0, n - frame + 1, hop)
    rms = np.empty(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        seg = samples[s:s + frame]
        rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2))
    times = (starts + frame / 2.0) / rate
    return times, rms


def _smooth(x, win):
    """Centered moving average with edge padding. win in samples."""
    win = max(1, int(win))
    if win == 1:
        return x.astype(np.float64)
    pad = win // 2
    xp = np.pad(x.astype(np.float64), pad, mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(xp, kernel, mode="same")[pad:pad + len(x)]


# A song only really ends after QUIET_S of quiet (Stefan 2026-11-10: "more like 10 seconds ... it
# doesn't have to be completely silent, just way more silent than a band. People may still talk
# into mics"). Quiet = below the song-start level (quiet_frac, same scale as hi_frac) once the level
# is smoothed with a QUIET_MEDIAN_S running median, so talking and crowd bursts drop out. Two
# windows with less quiet than that between them are one song with a quiet bit or a stop in it.
QUIET_S = 8.0           # 10 s joined two real songs at Litty D Misco 4 (a 9 s pause); 8 s gets all
                        # five shows with hand-fixed songs exactly right
QUIET_FRAC = 0.40
QUIET_MEDIAN_S = 3.0
QUIET_EDGE_S = 10.0      # the quiet may begin in a song's fading tail / end in the next one's count-in
MUSIC_DB = 8.0           # a piece whose typical level is this far under the band's (the show's 90th
                         # percentile) is someone talking or tuning into a mic, not a song


def _running_median(x, win):
    win = max(1, int(win) | 1)
    if win == 1 or len(x) < win:
        return x.astype(np.float64)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = win // 2
    return np.median(sliding_window_view(np.pad(x.astype(np.float64), pad, mode="edge"), win), axis=1)


def _longest_run(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _join_quiet_bits(segs, env_raw, thr_quiet, hop_s, quiet_s, band_db):
    """Merge neighbouring segments unless the stretch between them has a run of >= quiet_s below
    thr_quiet (on the running-median level)."""
    if quiet_s <= 0 or len(segs) < 2:
        return segs
    med = _running_median(env_raw, round(QUIET_MEDIAN_S / hop_s))
    need = int(round(quiet_s / hop_s))
    edge = int(round(QUIET_EDGE_S / hop_s))
    # a piece no louder than "quiet", or way under the band, is talk / tuning: it joins nothing
    music = max(thr_quiet, band_db - MUSIC_DB)
    segs = [(s, e) for s, e in segs if np.median(med[s:e + 1]) >= music] or segs
    out = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = out[-1]
        a, b = max(ps, pe - edge), min(e, s + edge)
        if _longest_run(med[a:b + 1] < thr_quiet) < need:
            out[-1] = (ps, e)
        else:
            out.append((s, e))
    # ...and the same after joining: bits of talk glued together are still talk
    return [(s, e) for s, e in out if np.median(med[s:e + 1]) >= music] or out


def detect_windows(rms, hop_s, smooth_s=1.5, min_gap_s=4.0, min_song_s=45.0,
                   hi_frac=0.40, lo_frac=0.25, min_stop_s=1.0, stop_frac=0.10,
                   quiet_s=QUIET_S, quiet_frac=QUIET_FRAC):
    """Segment a loudness envelope into song windows.

    `rms`   : per-frame RMS values (any positive loudness curve).
    `hop_s` : seconds between frames.
    smooth_s   : moving-average window over the loudness envelope.
    min_gap_s  : sustained SHALLOW-quiet (below thr_lo) needed to end a song — catches a long lull.
    min_stop_s : a shorter DEEP-quiet (below thr_stop, near the noise floor) that also ends a song —
                 catches a brief true STOP between songs without demanding the full min_gap. This is
                 what lets a ~1 s stop start a new song while a ~1 s musical hole (which only dips to
                 thr_lo, not thr_stop) does NOT falsely split one. Set it >= min_gap_s to disable.
    min_song_s : discard detected segments shorter than this (not full songs).
    hi/lo_frac : hysteresis thresholds as a fraction between the 10th and 90th
                 percentile of the smoothed loudness.
    stop_frac  : the deep-stop threshold as that same fraction (below lo_frac).
    quiet_s    : a song only ends where there is at least this much quiet (see QUIET_S); 0 = off.
    quiet_frac : "quiet" for that rule, as that same fraction.

    Returns (windows, thresholds) where each window is
      {index, start_s, end_s, dur_s, mean_db}
    and thresholds reports the dB band used. Deterministic.
    """
    rms = np.asarray(rms, dtype=np.float64)
    env_raw = 20.0 * np.log10(rms + 1e-6)               # un-smoothed: a brief true stop shows here
    env = _smooth(env_raw, round(smooth_s / hop_s))     # smoothed: stable lull / hysteresis detection

    floor = np.percentile(env, 10)
    ceil = np.percentile(env, 90)
    span = max(ceil - floor, 1e-6)
    thr_hi = floor + hi_frac * span
    thr_lo = floor + lo_frac * span
    thr_stop = floor + stop_frac * span

    segs = []
    in_song = False
    seg_start = 0
    low_run = 0            # consecutive frames below thr_lo (shallow)   -> the min_gap_s rule
    stop_run = 0           # consecutive frames below thr_stop (deep)    -> the min_stop_s rule
    min_gap_frames = int(round(min_gap_s / hop_s))
    min_stop_frames = max(1, int(round(min_stop_s / hop_s)))
    for i, v in enumerate(env):
        if not in_song:
            if v >= thr_hi:
                in_song = True
                seg_start = i
                low_run = 0
                stop_run = 0
        else:
            # Each in-song frame updates two independent runs: a SHALLOW run (smoothed env below
            # thr_lo) for the long-lull rule, and a DEEP run (RAW env below thr_stop) for a brief true
            # stop — the deep run is checked every frame so a ~1 s silence smoothing can't hide it.
            low_run = low_run + 1 if v < thr_lo else 0
            stop_run = stop_run + 1 if env_raw[i] < thr_stop else 0
            if stop_run >= min_stop_frames:            # brief deep stop (a true stop between songs)
                segs.append((seg_start, i - stop_run + 1))
                in_song = False
            elif low_run >= min_gap_frames:            # long shallow lull
                segs.append((seg_start, i - low_run + 1))   # ended when the low run began
                in_song = False
    if in_song:
        segs.append((seg_start, len(env) - 1))
    segs = _join_quiet_bits(segs, env_raw, floor + quiet_frac * span, hop_s, quiet_s, ceil)

    out = []
    for s, e in segs:
        start_s = s * hop_s
        end_s = min(e, len(env) - 1) * hop_s
        dur = end_s - start_s
        if dur < min_song_s:
            continue
        out.append({"index": len(out),
                    "start_s": round(float(start_s), 2),
                    "end_s": round(float(end_s), 2),
                    "dur_s": round(float(dur), 2),
                    "mean_db": round(float(np.mean(env[s:e + 1])), 2)})
    for i, d in enumerate(out):
        d["index"] = i
    thresholds = {"floor_db": round(float(floor), 2), "ceil_db": round(float(ceil), 2),
                  "thr_hi_db": round(float(thr_hi), 2), "thr_lo_db": round(float(thr_lo), 2)}
    return out, thresholds


# ----------------------------------------------------------------------------------
# Optional audio decode (ffmpeg) + one-call convenience.
# ----------------------------------------------------------------------------------
def _ffmpeg():
    for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"):
        if cand == "ffmpeg" or os.path.exists(cand):
            return cand
    return "ffmpeg"


def decode_mono(path, rate=DEFAULT_RATE):
    """Decode `path` to mono float32 at `rate` Hz via ffmpeg. Returns (samples, rate)."""
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", path, "-ac", "1", "-ar", str(rate),
         "-f", "f32le", "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype="<f4").copy(), rate


def split(path, rate=DEFAULT_RATE, frame_ms=DEFAULT_FRAME_MS, samples=None, **detect_kw):
    """Decode (or accept `samples`) and detect song windows. Returns
    (windows, thresholds, meta)."""
    if samples is None:
        samples, rate = decode_mono(path, rate=rate)
    _, rms = rms_envelope(samples, rate, frame_ms=frame_ms, hop_ms=frame_ms)
    windows, thr = detect_windows(rms, frame_ms / 1000.0, **detect_kw)
    meta = {"path": path, "rate": rate, "frame_ms": frame_ms,
            "duration_s": round(len(samples) / float(rate), 2) if len(samples) else 0.0,
            "count": len(windows)}
    return windows, thr, meta


def cut(path, windows, out_dir, reencode=False, ext=None, prefix="Song", names=None):
    """Export each detected window to its own file in `out_dir`, one per song. Works for ANY file
    ffmpeg reads — audio or video — because it copies whatever streams the input has.

    reencode=False (default): ffmpeg stream-copy (`-c copy`), lossless and fast; a video cut lands on
    the nearest keyframe at/just before each start. reencode=True: decode+re-encode for frame-accurate
    cuts (slower, and uses ffmpeg's default encoders for the container).

    `ext` overrides the output extension (default: the input's, so the type is preserved). Files are
    named "<prefix>_01.<ext>" by default; pass `names` (a list aligned to `windows`) to write
    "<name>.<ext>" instead — the app uses this to export "01 Black Heart.wav" etc. without touching
    the numbered working cuts. The input is never modified. Returns the list of written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_ext = ext or os.path.splitext(path)[1] or ".wav"
    if not out_ext.startswith("."):
        out_ext = "." + out_ext
    ff = _ffmpeg()
    made = []
    for i, w in enumerate(windows):
        start = float(w["start_s"])
        dur = max(0.01, float(w["end_s"]) - start)
        if names and i < len(names) and names[i]:
            out = os.path.join(out_dir, f"{names[i]}{out_ext}")
        else:
            out = os.path.join(out_dir, f"{prefix}_{w['index'] + 1:02d}{out_ext}")
        # -ss before -i = fast seek; -t = duration. Copy all streams unless re-encoding.
        cmd = [ff, "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", path, "-t", f"{dur:.3f}"]
        cmd += ([] if reencode else ["-c", "copy"]) + [out]
        subprocess.run(cmd, check=True, capture_output=True)
        made.append(out)
    return made


def fmt_ts(s):
    m, sec = divmod(int(round(s)), 60)
    return f"{m:02d}:{sec:02d}"


# ----------------------------------------------------------------------------------
# Self-test: synthesize a set of N songs with gaps and prove we recover N windows.
# ----------------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    rate = DEFAULT_RATE
    hop_ms = DEFAULT_FRAME_MS
    song_lens = [70, 55, 90, 60, 120]      # seconds
    gap_len = 8                            # seconds of applause/quiet between songs
    chunks = []
    for i, sl in enumerate(song_lens):
        # loud, dense music: strong noise, some dynamics but never gap-quiet.
        music = (rng.standard_normal(int(sl * rate)) * 0.6).astype(np.float32)
        chunks.append(music)
        if i < len(song_lens) - 1:
            quiet = (rng.standard_normal(int(gap_len * rate)) * 0.02).astype(np.float32)
            chunks.append(quiet)
    samples = np.concatenate(chunks)
    windows, thr, meta = split("<synthetic>", rate=rate, frame_ms=hop_ms, samples=samples)
    got = len(windows)
    want = len(song_lens)
    ok = (got == want)
    print(f"  synthesized {want} songs ({sum(song_lens)}s music, {gap_len}s gaps)")
    for w in windows:
        print(f"    #{w['index']}  {fmt_ts(w['start_s'])}–{fmt_ts(w['end_s'])}  "
              f"{w['dur_s']:>5.1f}s  {w['mean_db']:>6.1f} dB")
    print(f"  recovered {got}/{want}  thresholds={thr}")
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None):
    import json
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] == "selftest":
        return _selftest()

    path = argv[0]
    json_out = labels_out = None
    do_cut = False
    cut_dir = None
    reencode = False
    rest = argv[1:]
    i = 0
    while i < len(rest):
        if rest[i] == "--json" and i + 1 < len(rest):
            json_out = rest[i + 1]; i += 2
        elif rest[i] == "--labels" and i + 1 < len(rest):
            labels_out = rest[i + 1]; i += 2
        elif rest[i] == "--cut":
            do_cut = True; i += 1
            if i < len(rest) and not rest[i].startswith("--"):    # optional out_dir
                cut_dir = rest[i]; i += 1
        elif rest[i] == "--reencode":
            reencode = True; i += 1
        else:
            print("unknown arg:", rest[i]); return 2

    windows, thr, meta = split(path)
    print(f"thresholds: {thr}")
    print(f"{'#':>2}  {'start':>6}  {'end':>6}  {'dur':>6}  {'dB':>7}")
    for w in windows:
        print(f"{w['index']:>2}  {fmt_ts(w['start_s']):>6}  {fmt_ts(w['end_s']):>6}  "
              f"{w['dur_s']:>6.1f}  {w['mean_db']:>7.2f}")
    print(f"\n{meta['count']} songs over {fmt_ts(meta['duration_s'])}")

    if json_out:
        with open(json_out, "w") as f:
            json.dump({"meta": meta, "thresholds": thr, "songs": windows}, f, indent=2)
        print("wrote", json_out)
    if labels_out:
        # Audacity-style label track: start\tend\tname (tab-separated), importable elsewhere too.
        with open(labels_out, "w") as f:
            for w in windows:
                f.write(f"{w['start_s']:.2f}\t{w['end_s']:.2f}\tSong {w['index'] + 1}\n")
        print("wrote", labels_out)
    if do_cut:
        # Default output: "<input> songs/" beside the input, so pieces don't clutter the source.
        if cut_dir is None:
            base = os.path.splitext(os.path.basename(path))[0]
            cut_dir = os.path.join(os.path.dirname(os.path.abspath(path)), f"{base} songs")
        if not windows:
            print("nothing to cut — no songs detected")
        else:
            made = cut(path, windows, cut_dir, reencode=reencode)
            print(f"cut {len(made)} song{'s' if len(made) != 1 else ''} "
                  f"({'re-encoded' if reencode else 'stream-copied'}) into {cut_dir}")
            for p in made:
                print("  ", os.path.basename(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
