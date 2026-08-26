"""Distribution preservation. The emitted token must follow probs."""
import collections
import random
import sys
import numpy as np
from scipy import stats
from util import one
from stego.reg import get, names

TRIALS = 40000


def check(name, p, idx):
    c = get(name)
    cnt = collections.Counter()
    src = random.Random(999)
    for _ in range(TRIALS):
        rng = random.Random(src.getrandbits(64))
        bits = "".join(src.choice("01") for _ in range(64))
        tid, _, _ = c.encode_step(p, idx, bits, c.new_state(), rng)
        cnt[tid] += 1
    obs = np.array([cnt.get(i, 0) for i in idx], dtype=float)
    chi2, pv = stats.chisquare(obs, np.array(p) * TRIALS)
    tvd = 0.5 * np.abs(obs / TRIALS - np.array(p)).sum()
    return chi2, pv, tvd


def main():
    p, idx = one(12, 5)
    good = True
    print("{:9s} {:>10s} {:>9s} {:>8s}".format("codec", "chi2", "p", "tvd"))
    for n in names():
        chi2, pv, tvd = check(n, p, idx)
        ok = pv > 0.01
        good &= ok
        print("{:9s} {:10.2f} {:9.4f} {:8.4f}  {}".format(n, chi2, pv, tvd, "ok" if ok else "BIASED"))
    sys.exit(0 if good else 1)


main()
