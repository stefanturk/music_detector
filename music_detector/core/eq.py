"""EQ ownership: a mic is really being played (or sung into) when it OWNS its instrument's frequency
range -- its energy there stands well above what the other mics pick up in the same range,
compared with its own resting level of bleed.

Loudness alone can't tell a singer from the band bleeding into her mic: study S2 (Litty D Misco 4,
2026-11-09) scored loudness 0.38 F1 against Stefan's corrected lyric lines and loudness AND vocal
presence-range (2-5 kHz) ownership 0.74. But who's-playing's own bleed rules already get there
(0.792 F1 on all 11 songs with or without this), so it is NOT wired into the analysis. It's kept
for the horn mics (1-4 kHz), where a sax mic hearing the trumpet still counts as playing -- that
waits for Stefan's ear check. Deterministic numpy; when the audio can't be read, None.
"""
import os
import re
import subprocess

import numpy as np

RATE = 16000
BIN_S = 0.5                 # ownership is judged per half second
BANDS = {"VOCAL": (2000.0, 5000.0), "SAX": (1000.0, 4000.0), "TRUMPET": (1000.0, 4000.0)}
ABOVE_DB = 10.0             # this far above the mic's resting ratio (its 20th percentile)
REST_PCT = 20
NOT_BAND = re.compile(r"\b(TOM|TOMS)\b", re.I)


def _decode(path, start, dur, ffmpeg):
    p = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}",
                        "-i", path, "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
                       capture_output=True)
    if p.returncode != 0:
        return None
    x = np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)
    if start < 0:
        x = np.concatenate([np.zeros(int(-start * RATE)), x])
    return x


def band_bins(x, lo, hi, bin_s=BIN_S):
    """Energy in [lo, hi] Hz per bin."""
    per = int(bin_s * RATE)
    n = len(x) // per
    if n == 0:
        return np.zeros(0)
    X = x[:n * per].reshape(n, per)
    S = np.abs(np.fft.rfft(X * np.hanning(per), axis=1)) ** 2
    f = np.fft.rfftfreq(per, 1.0 / RATE)
    return S[:, (f >= lo) & (f <= hi)].sum(axis=1)


def owned(mic_bins, others_bins, above_db=ABOVE_DB):
    """Per bin: does this mic own the range (well above its own resting ratio to the others)?"""
    m = min(len(mic_bins), len(others_bins))
    if m == 0:
        return np.zeros(0, dtype=bool)
    r = 10 * np.log10(mic_bins[:m] + 1e-12) - 10 * np.log10(others_bins[:m] + 1e-12)
    return r > np.percentile(r, REST_PCT) + above_db


def voice_ownership(voices, stem_dir, start_s, dur_s, n_frames, hop, voice="VOCAL", ffmpeg=None):
    """Per frame (hop seconds, n_frames long): does any of `voice`'s mics own its range against
    every other instrument mic? None when it can't be judged (no such voice, no audio)."""
    lo_hi = BANDS.get(voice)
    files = (voices or {}).get(voice) or []
    others = [f for v, fs in (voices or {}).items() if v != voice for f in fs
              if not NOT_BAND.search(os.path.basename(f))]
    if not lo_hi or not files or not others:
        return None
    ffmpeg = ffmpeg or os.environ.get("MUSIC_DETECTOR_FFMPEG", "/opt/homebrew/bin/ffmpeg")
    if not os.path.exists(ffmpeg):
        return None
    try:
        ob = None
        for f in others:
            x = _decode(os.path.join(stem_dir, f), start_s, dur_s, ffmpeg)
            if x is None:
                return None
            b = band_bins(x, *lo_hi)
            ob = b if ob is None else ob[:len(b)] + b[:len(ob)]
        own = None
        for f in files:
            x = _decode(os.path.join(stem_dir, f), start_s, dur_s, ffmpeg)
            if x is None:
                return None
            b = band_bins(x, *lo_hi)
            if not len(b) or np.percentile(b, 95) < 1e-10:
                continue                                   # this mic is off for the song
            o = owned(b, ob)
            own = o if own is None else own[:len(o)] | o[:len(own)]
    except Exception:
        return None
    if own is None:
        return None
    per = max(1, int(round(BIN_S / hop)))
    frames = np.repeat(own, per)[:n_frames]
    if len(frames) < n_frames:
        frames = np.concatenate([frames, np.zeros(n_frames - len(frames), dtype=bool)])
    return frames
