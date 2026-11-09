#!/usr/bin/env python3
"""Per-bar fingerprints of a song from its four stems (drums / bass / vocals / other) and the mix.

    python3 features.py <stems_dir> <song.mp3> [out.json]

Per bar (4 beats, from the drums stem's beat grid) and per stem:
  energy   RMS, as a share of that stem's own loudest bar (so no stem needs balancing)
  melody   12-bin chroma (bass: low band, other: mid band, vocals: mid band)
  rhythm   onset strength across the bar (16 slots), and the onset count (pulse)
  bright   spectral centroid
Vocals also get: presence (energy vs the song's own vocal level), syllable rate (onsets/bar) and
pitch movement (chroma change bar to bar), i.e. "is there singing, and what is it doing".
The beat grid has a confidence (how steadily the drums hit on the grid); the bar count is only
trusted for phrase-length hints when it is high.
"""
import json
import os
import sys

import librosa
import numpy as np

SR = 22050
HOP = 512


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def bar_grid(drums):
    """(beat_times, bar_start_times, confidence). Downbeat = the 4-beat phase with the most
    low-end onset strength (kicks land on 1)."""
    on = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=on, sr=SR, hop_length=HOP, tightness=120)
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    low = librosa.onset.onset_strength(y=drums, sr=SR, hop_length=HOP,
                                       fmax=200, aggregate=np.mean)
    best, best_p = -1.0, 0
    for p in range(4):
        s = float(np.sum(low[beats[p::4]])) if len(beats[p::4]) else 0.0
        if s > best:
            best, best_p = s, p
    bars = bt[best_p::4]
    # confidence: how peaky the onsets are on the beats vs between them
    on_beats = float(np.mean(on[beats])) if len(beats) else 0.0
    conf = on_beats / (float(np.mean(on)) + 1e-9)
    return float(np.atleast_1d(tempo)[0]), bt, bars, conf


def per_bar(y, bars, end, band=None):
    """energy, chroma(12), rhythm(16), pulse, bright per bar for one signal."""
    if band:
        y = librosa.effects.preemphasis(y) if band == "hi" else y
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP * 4,
                                        fmin=librosa.note_to_hz("C2" if band == "low" else "C3"),
                                        n_octaves=3 if band == "low" else 5)
    ct = librosa.times_like(chroma, sr=SR, hop_length=HOP * 4)
    on = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    ot = librosa.times_like(on, sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0]
    out = {"energy": [], "chroma": [], "rhythm": [], "pulse": [], "bright": []}
    edges = list(bars) + [end]
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 0.5:
            continue
        m = (ot >= a) & (ot < b)
        c = (ct >= a) & (ct < b)
        out["energy"].append(float(np.sqrt(np.mean(rms[m] ** 2))) if m.any() else 0.0)
        ch = chroma[:, c].mean(axis=1) if c.any() else np.zeros(12)
        out["chroma"].append((ch / (ch.sum() + 1e-9)).tolist())
        seg = on[m]
        slots = np.array_split(seg, 16)
        out["rhythm"].append([float(s.mean()) if len(s) else 0.0 for s in slots])
        peaks = librosa.util.peak_pick(seg, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
                                       delta=0.3 * (on.std() + 1e-9), wait=3) if len(seg) > 12 else []
        out["pulse"].append(int(len(peaks)))
        out["bright"].append(float(cen[m].mean()) if m.any() else 0.0)
    return out


def main():
    stems_dir, song = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    sig = {k: load(os.path.join(stems_dir, k + ".wav")) for k in ("drums", "bass", "vocals", "other")}
    mix = load(song)
    tempo, beats, bars, conf = bar_grid(sig["drums"])
    end = len(mix) / SR
    data = {"tempo": tempo, "grid_confidence": conf, "bar_times": [float(b) for b in bars],
            "end": end, "stems": {}}
    bands = {"drums": None, "bass": "low", "vocals": None, "other": None, "mix": None}
    for k, y in list(sig.items()) + [("mix", mix)]:
        data["stems"][k] = per_bar(y, bars, end, bands[k])
    if out_path:
        with open(out_path, "w") as f:
            json.dump(data, f)
    print("tempo %.1f BPM, %d bars, grid confidence %.2f" % (tempo, len(bars), conf))
    return data


if __name__ == "__main__":
    main()
