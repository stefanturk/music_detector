"""What kind of sound a part is making, from how it behaves, not from a trained model.

Takes one isolated part (a demucs split part or a real mic, mono) and measures, every BIN_S seconds:
  level     loudness (dB, relative to the part's own loud level)
  clarity   how clearly ONE pitch is heard (YIN): a horn line or a lead line is high; a strummed
            chord, a pad or noise is low
  sustain   how much of a note is still there SUSTAIN_AT after it starts: plucked/struck notes
            (guitar, piano) die away; blown notes (horns) and organ/pads hold or swell
  vibrato   share of held notes with a 4-8 Hz pitch wobble of 10-80 cents (players and singers do
            it; organ, synth pads and guitar chords don't)
  bark      share of the energy at 1-3 kHz, where brass and sax are loud
  bright    share of the energy above 2 kHz
  flat      spectral flatness (noise-like vs tonal)

Pure numpy, deterministic. The thresholds in `horn_likeness` come from how the instruments make
sound, not from fitting any one band. The same measurements feed lead-vs-rhythm, held vocal notes
and sung-vs-spoken later.
"""
import numpy as np

BIN_S = 0.5               # same bins as the energy / featured curves
FRAME = 1024              # analysis frame (samples at RATE)
RATE = 22050
HOP = 512                 # ~23 ms
FMIN, FMAX = 80.0, 1100.0  # horn, guitar-lead and vocal fundamentals
YIN_TH = 0.15

SUSTAIN_AT = 0.25         # seconds after a note starts
ONSET_RISE_DB = 6.0       # a note starts where the level jumps this much in ONSET_WIN
ONSET_WIN = 0.07
NOTE_MIN_S = 0.30         # held notes this long are checked for vibrato
VIB_HZ = (4.0, 8.0)
VIB_CENTS = (10.0, 80.0)
VIB_CHUNK_S = 1.0
CONTEXT_S = 2.0           # note-based measures are pooled over this window around each bin
QUIET_DB = -30.0
HIGHPASS_HZ = 150.0
SILENT_DBFS = -70.0       # and never below this, whatever the part's own level          # frames this far under the part's loud level are "not playing"


def _frames(x, frame=FRAME, hop=HOP):
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx] if n and len(x) >= frame else np.zeros((0, frame))


def yin(frames, rate=RATE, fmin=FMIN, fmax=FMAX, th=YIN_TH):
    """(f0 Hz, clarity 0..1) per frame. f0 is NaN where no clear pitch is heard."""
    n, w = frames.shape
    tmax = min(int(rate / fmin), w // 2)
    tmin = max(2, int(rate / fmax))
    if n == 0:
        return np.zeros(0), np.zeros(0)
    half = w // 2
    x = frames - frames.mean(axis=1, keepdims=True)
    # difference function d(tau) = sum (x[j] - x[j+tau])^2 over j < half, via FFT correlation
    F = np.fft.rfft(x, 2 * w)
    seg = np.fft.rfft(x[:, :half], 2 * w)
    corr = np.fft.irfft(F * np.conj(seg), 2 * w)[:, :tmax + 1]
    sq = np.cumsum(np.concatenate([np.zeros((n, 1)), x ** 2], axis=1), axis=1)
    e0 = sq[:, half][:, None]
    taus = np.arange(tmax + 1)
    et = sq[:, taus + half] - sq[:, taus]
    d = e0 + et - 2 * corr
    d[:, 0] = 0
    cs = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, tmax + 1)[None, :] / (cs + 1e-12)
    sub = cmnd[:, tmin:tmax]
    f0 = np.full(n, np.nan)
    clar = np.zeros(n)
    below = sub < th
    first = np.where(below.any(axis=1), below.argmax(axis=1), -1)
    for i in range(n):
        if first[i] >= 0:
            k = first[i]
            while k + 1 < sub.shape[1] and sub[i, k + 1] < sub[i, k]:
                k += 1
        else:
            k = int(np.argmin(sub[i]))
        v = sub[i, k]
        clar[i] = float(np.clip(1.0 - v, 0.0, 1.0))
        if v < th:
            t = k + tmin
            if 1 <= k < sub.shape[1] - 1:      # parabolic refinement
                a, b, c = sub[i, k - 1], sub[i, k], sub[i, k + 1]
                den = a - 2 * b + c
                t = t + (0.5 * (a - c) / den if den else 0.0)
            f0[i] = rate / t
    return f0, clar


def _notes(f0, voiced, fps):
    """Runs of voiced frames that stay within a semitone of their own median: (start, end) frames."""
    out = []
    n = len(f0)
    i = 0
    min_len = int(NOTE_MIN_S * fps)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        ref = f0[i]
        while j + 1 < n and voiced[j + 1] and abs(1200 * np.log2(f0[j + 1] / ref)) < 100:
            j += 1
            ref = np.median(f0[i:j + 1])
        if j + 1 - i >= min_len:
            out.append((i, j + 1))
        i = j + 1
    return out


def _vibrato(cents, fps):
    """True when a held note's pitch wobbles at 4-8 Hz by 10-80 cents."""
    c = np.asarray(cents, dtype=float)
    t = np.arange(len(c))
    if len(c) < 4:
        return False
    p = np.polyfit(t, c, 1)
    c = c - np.polyval(p, t)
    spec = np.abs(np.fft.rfft(c * np.hanning(len(c)))) ** 2
    f = np.fft.rfftfreq(len(c), 1.0 / fps)
    if len(spec) < 3:
        return False
    band = (f >= VIB_HZ[0]) & (f <= VIB_HZ[1])
    if not band.any():
        return False
    share = spec[band].sum() / (spec[1:].sum() + 1e-12)
    extent = 2 * np.sqrt(2) * np.std(c)           # peak-to-peak of a sine with this std
    return bool(share > 0.4 and VIB_CENTS[0] <= extent <= VIB_CENTS[1] * 2)


def _highpass(x, rate, hz=HIGHPASS_HZ):
    """Drop kick / bass rumble (it swamps the pitch and tone measures on a close mic and leaks into
    split parts). Horn, lead and vocal pitch is still found from the harmonics above."""
    from scipy.signal import butter, sosfiltfilt
    if len(x) < 64:
        return x
    return sosfiltfilt(butter(4, hz, "highpass", fs=rate, output="sos"), x)


def measure(x, rate=RATE, n_bins=None, bin_s=BIN_S):
    """Per-bin measurements of one part: {name: np.array(n_bins)}. `x` is mono float at `rate`."""
    x = _highpass(np.asarray(x, dtype=np.float64), rate)
    fr = _frames(x)
    fps = rate / HOP
    nb = n_bins if n_bins is not None else int(np.ceil(len(x) / rate / bin_s))
    out = {k: np.zeros(nb) for k in ("level", "clarity", "sustain", "vibrato", "bark", "bright",
                                     "flat", "playing")}
    if len(fr) == 0 or nb == 0:
        return out
    win = np.hanning(FRAME)
    spec = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
    f = np.fft.rfftfreq(FRAME, 1.0 / rate)
    tot = spec.sum(axis=1) + 1e-12
    rms = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    loud = np.percentile(db, 95)
    rel = db - loud
    playing = (rel > QUIET_DB) & (db > SILENT_DBFS)
    bark = spec[:, (f >= 1000) & (f <= 3000)].sum(axis=1) / tot
    bright = spec[:, f >= 2000].sum(axis=1) / tot
    sp = spec[:, (f >= 100) & (f <= 6000)] + 1e-12
    flat = np.exp(np.mean(np.log(sp), axis=1)) / np.mean(sp, axis=1)
    f0, clar = yin(fr, rate)
    voiced = ~np.isnan(f0) & playing

    # note starts: level jumps ONSET_RISE_DB within ONSET_WIN, then sustain = level SUSTAIN_AT later
    k = max(1, int(round(ONSET_WIN * fps)))
    lag = max(1, int(round(SUSTAIN_AT * fps)))
    on_t, on_sus = [], []
    i = k
    while i < len(db) - lag:
        if playing[i] and db[i] - db[i - k] >= ONSET_RISE_DB:
            pk = i + int(np.argmax(db[i:i + k + 1]))
            on_t.append(pk)
            on_sus.append(float(np.clip(10 ** ((db[min(len(db) - 1, pk + lag)] - db[pk]) / 20), 0, 1.5)))
            i = pk + k
        else:
            i += 1
    on_t = np.asarray(on_t)
    on_sus = np.asarray(on_sus)

    # vibrato, judged on held notes in chunks of up to VIB_CHUNK_S (a note can start straight and
    # bloom); each frame of a judged chunk gets 1 (wobbles) or 0 (straight), others stay NaN
    vib = np.full(len(f0), np.nan)
    chunk = int(VIB_CHUNK_S * fps)
    min_len = int(NOTE_MIN_S * fps)
    for a, b in _notes(f0, voiced, fps):
        for c0 in range(a, b, chunk):
            c1 = min(b, c0 + chunk)
            if c1 - c0 < min_len:
                if c0 > a:                       # a short tail joins the chunk before it
                    vib[c0:c1] = vib[c0 - 1]
                continue
            cents = 1200 * np.log2(f0[c0:c1] / np.median(f0[c0:c1]))
            vib[c0:c1] = 1.0 if _vibrato(cents, fps) else 0.0

    per = int(round(bin_s * fps))
    ctx = CONTEXT_S * fps / 2
    for b in range(nb):
        a0, a1 = b * per, min(len(db), (b + 1) * per)
        if a0 >= len(db):
            break
        sl = slice(a0, a1)
        pl = playing[sl]
        out["playing"][b] = float(pl.mean()) if a1 > a0 else 0.0
        out["level"][b] = float(np.max(rel[sl])) if a1 > a0 else -99.0
        if not pl.any():
            continue
        out["clarity"][b] = float(np.median(clar[sl][pl]))
        out["bark"][b] = float(np.median(bark[sl][pl]))
        out["bright"][b] = float(np.median(bright[sl][pl]))
        out["flat"][b] = float(np.median(flat[sl][pl]))
        mid = (a0 + a1) / 2
        if len(on_t):
            near = np.abs(on_t - mid) <= ctx
            out["sustain"][b] = float(np.median(on_sus[near])) if near.any() else np.nan
        else:
            out["sustain"][b] = np.nan
        v = vib[max(0, int(mid - ctx)):int(mid + ctx)]
        v = v[~np.isnan(v)]
        out["vibrato"][b] = float(v.mean()) if len(v) else 0.0
    # a bin with no note start nearby is one long held sound
    s = out["sustain"]
    s[np.isnan(s)] = 1.0
    return out


def horn_likeness(m):
    """0..1 per bin: does this sound like a horn (or horn section)? Held notes and a clear pitch,
    helped by vibrato and a 1-3 kHz bark; plucked notes and noise count against it."""
    held = np.clip((m["sustain"] - 0.45) / 0.35, 0, 1)
    pitch = np.clip((m["clarity"] - 0.55) / 0.3, 0, 1)
    bark = np.clip((m["bark"] - 0.08) / 0.25, 0, 1)
    tonal = np.clip((0.35 - m["flat"]) / 0.25, 0, 1)
    s = 0.35 * held + 0.3 * pitch + 0.15 * m["vibrato"] + 0.2 * bark
    return np.where(m["playing"] > 0.2, s * tonal, 0.0)
