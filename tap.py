#!/usr/bin/env python3
"""Label a song's sections by tapping along. Plays the file and records the time of every
Enter you press: press it the moment you HEAR the part change (verse -> chorus, a drop, the
band dropping out, a new groove). Typing a short word before Enter names the new part.

    python3 tap.py "<song>.mp3" [--who user|claude]

Saved to ~/Music/Beat Thief Analysis/labels/<song>.<who>.json. A person taps a little late
(about a quarter of a second); the raw times are kept and the lag is corrected when scoring.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

CACHE = os.path.join(os.path.expanduser("~"), "Music", "Beat Thief Analysis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--who", default="user")
    a = ap.parse_args()
    song = os.path.abspath(a.song)
    name = os.path.splitext(os.path.basename(song))[0]
    out_dir = os.path.join(CACHE, "labels")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "%s.%s.json" % (name, a.who))
    if os.path.exists(out):
        print("Already labelled (%s). Delete it to redo." % out)
        sys.exit(1)

    print("\n%s\n" % name)
    print("Press ENTER each time the part CHANGES. Type a word first to name it (optional).")
    print("The song starts after 3 seconds. Ctrl-C saves what you have.\n")
    for n in (3, 2, 1):
        print(n, end=" ", flush=True)
        time.sleep(1)
    print("\n")

    cmd = ["afplay", song]
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd)
    taps, done = [], threading.Event()

    def watch():
        proc.wait()
        done.set()
        print("\n-- song over: press ENTER once to save --")
    threading.Thread(target=watch, daemon=True).start()
    try:
        while True:
            word = input()
            t = time.monotonic() - t0
            if done.is_set():
                break
            taps.append({"t": round(t, 2), "name": word.strip()})
            print("   %5.1fs  %s" % (t, word.strip() or "(change)"))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
    with open(out, "w") as f:
        json.dump({"song": name, "who": a.who, "tap_lag_s": 0.25,
                   "changes": taps}, f, indent=1)
    print("\nSaved %d changes to %s" % (len(taps), out))


if __name__ == "__main__":
    main()
