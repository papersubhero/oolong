"""n_bits must mean the same thing in every codec: payload consumed."""
import random
import sys
from util import dists
from stego.reg import get, names


def main():
    D = dists(100, 64, 7)
    good = True
    print("{:9s} {:>18s}".format("codec", "n_bits, no payload"))
    for n in names():
        c = get(n)
        rng, st, tot = random.Random(3), c.new_state(), 0
        for p, i in D:
            _, nb, st = c.encode_step(p, i, "", st, rng)
            tot += nb
        good &= tot == 0
        print("{:9s} {:18d}  {}".format(n, tot, "ok" if tot == 0 else "COUNTS CAPACITY"))
    sys.exit(0 if good else 1)


main()
