#!/usr/bin/env python3
"""Make 4-stem Demucs stems and per-bar features for every MP3 in ~/Music/Beat Thief.

    python3 pipeline.py [--budget-min 45] [--limit N]

Resumable and safe to run again and again (the hourly cron does): a song with stems is skipped, a
song with features is skipped, and a killed run leaves no half-done song (Demucs writes to a temp
folder that is renamed only when finished). The Beat Thief folders are only READ. Everything it
writes goes under ~/Music/Beat Thief Analysis/{stems,features,logs}. No new song is started after
the time budget, so one firing never runs into the next.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
LIB = os.path.join(HOME, "Music", "Beat Thief")
BASE = os.path.join(HOME, "Music", "Beat Thief Analysis")
HERE = os.path.dirname(os.path.abspath(__file__))
STEMS = ("drums", "bass", "vocals", "other")
ALIAS = {  # songs labelled earlier keep their short feature file names
    "You Are The Universe - The Brand New Heavies (125.5 BPM)": "universe",
    "Taste Of You - Allen Stone (131.9 BPM)": "taste",
}


def songs():
    out = {}
    for p in sorted(glob.glob(os.path.join(LIB, "**", "*.mp3"), recursive=True)):
        out.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return out


def stem_dir(name):
    return os.path.join(BASE, "stems", name)


def feat_path(name):
    return os.path.join(BASE, "features", ALIAS.get(name, name) + ".json")


def have_stems(name):
    return all(os.path.exists(os.path.join(stem_dir(name), s + ".wav")) for s in STEMS)


def make_stems(name, mp3):
    tmp = os.path.join(BASE, "stems", ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    log = os.path.join(BASE, "logs", "demucs_all.log")
    with open(log, "ab") as lf:
        lf.write(("\n=== %s ===\n" % name).encode())
        r = subprocess.run(["/usr/bin/python3", "-m", "demucs", "-n", "htdemucs", "-o", tmp, mp3],
                           stdout=lf, stderr=lf)
    if r.returncode != 0:
        raise RuntimeError("demucs failed (%d)" % r.returncode)
    made = glob.glob(os.path.join(tmp, "htdemucs", "*"))
    if not made or not all(os.path.exists(os.path.join(made[0], s + ".wav")) for s in STEMS):
        raise RuntimeError("demucs output incomplete")
    os.makedirs(os.path.dirname(stem_dir(name)), exist_ok=True)
    os.rename(made[0], stem_dir(name))
    shutil.rmtree(tmp, ignore_errors=True)


def make_features(name, mp3):
    os.makedirs(os.path.dirname(feat_path(name)), exist_ok=True)
    part = feat_path(name) + ".part"
    r = subprocess.run(["/usr/bin/python3", os.path.join(HERE, "features.py"), stem_dir(name), mp3, part],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(part):
        raise RuntimeError("features failed: " + r.stderr[-300:])
    os.rename(part, feat_path(name))
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=45)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    t0, lib = time.time(), songs()
    done_now, failed = 0, []
    bad_path = os.path.join(BASE, "logs", "failed.json")
    bad = json.load(open(bad_path)) if os.path.exists(bad_path) else {}
    for name, mp3 in lib.items():
        if a.limit and done_now >= a.limit:
            break
        if bad.get(name, 0) >= 2:
            continue                                  # gave up after two failures
        if os.path.exists(feat_path(name)) and have_stems(name):
            continue
        if time.time() - t0 > a.budget_min * 60:
            break                                     # out of time; the next firing carries on
        try:
            s = time.time()
            if not have_stems(name):
                make_stems(name, mp3)
            info = make_features(name, mp3)
            done_now += 1
            print("done  %-60s %4.0fs  %s" % (name[:60], time.time() - s, info), flush=True)
        except Exception as e:
            bad[name] = bad.get(name, 0) + 1
            failed.append(name)
            print("FAIL  %s: %s" % (name, e), flush=True)
    json.dump(bad, open(bad_path, "w"))
    left = sum(1 for n in lib if not (os.path.exists(feat_path(n)) and have_stems(n)) and bad.get(n, 0) < 2)
    marker = os.path.join(BASE, "all_done")
    if left == 0:
        open(marker, "w").write(time.strftime("%F %T"))
    elif os.path.exists(marker):
        os.remove(marker)
    print("finished %d this run, %d failed, %d left of %d" % (done_now, len(failed), left, len(lib)))


if __name__ == "__main__":
    main()
