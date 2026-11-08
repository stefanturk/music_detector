#!/usr/bin/env python3
"""How well two label files agree.  python3 agree.py <song name> [a=user] [b=claude]

Each side's changes are matched one-to-one to the nearest change of the other within a
tolerance (the tapper's lag is subtracted first). Prints F1 at +-2/4/8 s, what matched, what
only one side marked, and the average signed offset of the matches (an estimate of tap lag).
"""
import json, os, sys
LAB = os.path.join(os.path.expanduser("~"), "Music", "Beat Thief Analysis", "labels")

def load(name, who):
    d = json.load(open(os.path.join(LAB, "%s.%s.json" % (name, who))))
    lag = d.get("tap_lag_s", 0.0)
    return [(c["t"] - lag, c.get("name", ""), c.get("strength", 1.0)) for c in d["changes"]]

def match(a, b, tol):
    used, pairs = set(), []
    for i, (t, *_r) in enumerate(a):
        best = None
        for j, (s, *_q) in enumerate(b):
            if j in used or abs(s - t) > tol:
                continue
            if best is None or abs(s - t) < abs(b[best][0] - t):
                best = j
        if best is not None:
            used.add(best); pairs.append((i, best))
    return pairs

def f1(h, na, nb):
    p, r = (h / nb if nb else 0), (h / na if na else 0)
    return 2 * p * r / (p + r) if p + r else 0.0

def main():
    name = sys.argv[1]
    wa = sys.argv[2] if len(sys.argv) > 2 else "user"
    wb = sys.argv[3] if len(sys.argv) > 3 else "claude"
    A, B = load(name, wa), load(name, wb)
    Bmaj = [x for x in B if x[2] >= 1.0]
    print("%s: %d %s changes, %d %s (%d section-level)" % (name, len(A), wa, len(B), wb, len(Bmaj)))
    for label, BB in (("all %s" % wb, B), ("%s section-level" % wb, Bmaj)):
        for tol in (2.0, 4.0, 8.0):
            h = len(match(A, BB, tol))
            print("  %-22s +-%ds: matched %2d  F1 %.2f" % (label, tol, h, f1(h, len(A), len(BB))))
    pairs = match(A, B, 4.0)
    ma = {i for i, _ in pairs}; mb = {j for _, j in pairs}
    print("matched (+-4 s):")
    for i, j in pairs:
        print("  %6.1f %-12s ~ %6.1f %s  (%+.1f s)" % (A[i][0], A[i][1][:12], B[j][0], B[j][1][:40], A[i][0] - B[j][0]))
    print("only %s:" % wa, ["%.1f %s" % (A[i][0], A[i][1][:14]) for i in range(len(A)) if i not in ma])
    print("only %s:" % wb, ["%.1f%s" % (B[j][0], "" if B[j][2] >= 1 else "*") for j in range(len(B)) if j not in mb], "(* = phrase-level)")
    if pairs:
        print("average offset %s - %s on matches: %+.2f s" % (wa, wb, sum(A[i][0] - B[j][0] for i, j in pairs) / len(pairs)))

if __name__ == "__main__":
    main()
