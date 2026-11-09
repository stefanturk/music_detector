# music_detector

What the band is doing, read from a multitrack recording. A shared, app-agnostic core: movie_maker
uses it today; stage_lighting can use it later (live mode).

## Core package (`music_detector/core/`) — studio mode
Install editable into the app's Python: `pip install -e ../music_detector` (numpy only).
- `whoplays.py` who plays when (moved from movie_maker's fcpkit/whoplays.py; same numbers): per-voice
  50 ms envelopes, vocal hysteresis, lead labels in 2 s chunks. Ties in the label smoothing now keep
  the frame's own label (the old `max(set(...))` depended on Python's per-run hash seed).
- `detectors.py` playing on/off, featured score (share of the band vs usual, the keys "space"
  rule), tom fills (fallback from drum bursts when there are no tom stems), solos, trading,
  entrances, which vocal mic sings each lyric line, the energy curve.
- `studio.py` `detect()` runs them all on one song -> one dict (schema documented at the top);
  `summary()` prints it.
- Tests: `python3 -m pytest tests` (synthetic stems, one per detector).

Live mode (streaming the same detectors, for stage_lighting) is not built yet.

## Song export (`music_detector.songs`)

Cuts a show's one long recording into its songs (`01 - Black Heart.wav` + `.mp3`, 50 ms fades). Movie Maker and Ableton Loader share it. The folder's `.song_export.json` manifest means only songs whose window changed render again, songs that are gone get their files removed, and apps can tell the folder is an export, not show media.

## Workbench (`workbench/`) — shelved section-change experiments
Shelved 2026-11-07. An offline Mac workbench for "did the music just change section?", built to
drive Auto Scene's "On Change" in `stage_lighting`. Not good enough yet, so it was taken out of
stage_lighting (removed there, commit message mentions it). Nothing here runs on the Pi.

Goal: change scenes WHEN the part changes (about 2 s after, like a human notices), from separate
mixer channels (live band) or from one full-mix line-in (a DJ).

### Pieces
- `pipeline.py` Demucs 4-stem split of every song in `~/Music/Beat Thief/*/`; `run_hourly.sh` is the cron wrapper.
- `features.py` per-bar fingerprints per stem and mix (energy, chroma, rhythm, pulse, brightness).
- `detect.py` causal vote detector, modes "stems" and "mix"; scored by F1 at +-4 s. `--tune`, `--show <song>`.
- `multitrack.py`, `live.py` per-channel features and detector for a real multitrack (Litty D, Misco 4):
  channel ON/OFF arrangement events, plus section changes from per-channel chroma novelty.
- `tap.py`, `table.py`, `agree.py` ground truth: the user taps changes while the song plays (`tap.py`), I
  label blind from `table.py` tables, `agree.py` scores. `labels/` holds those files.
- `report.py` chart page and the fake-DJ-set splice test.
- Cache (10 GB, regenerable, NOT in git): `~/Music/Beat Thief Analysis/{stems,features,reports,multitrack}`.

### Scores (F1, +-4 s)
- Library songs, untuned defaults, vs the user's taps, stems / mix: 0.74/0.67, 0.67/0.62, 0.63/0.69, 0.67/0.53, 0.60/0.48.
- Splice test (hard-cut fake DJ sets): stems 0.71, mix 0.65.
- Multitrack, tuned on both songs (in-sample): song A 0.65, song B 0.62. Tuned on A alone: 0.80 on A, 0.32 on B.
  Never tested on a fresh stretch of the set.

### What we learned
- The user's taps are noisy: user vs me agree only 0.54-0.82, so ~0.75 is the practical ceiling.
- A one-line mix hides most changes: only ~6 of 15 in-song band changes show as a >=3 dB step in the summed mix.
- Latency is the real problem: ~3-4 s after the change (3 s window + 2 s peak check); On Change arrangement events wait 7 s.
  F1 at +-4 s hides that. Idea: fast path (a channel enters, ~1-1.5 s) plus slow path (subtle shifts), and snap to the next downbeat.
- Bar grids are sometimes wrong, not just half/double time (JB's Out! 91.5 -> 135, Give You Blue 150 -> 75): use windows in seconds.
- Vox entries are poorly caught by pitch novelty alone; they need the who's-playing counts.

## If revived
1. Measure latency per tap (seconds after the user's tap that the detector fires).
2. Test `live.py` untouched on a different 15 min of the Misco 4 set.
3. Stems detector: persistence (~2 bars), 4/8-bar snap, tempo-jump votes for DJ mixes, crossfade splice test, seconds-based windows, "entry" rule.
4. Port behind a single function in the Scene engine; check numpy and CPU on the Pi first.
