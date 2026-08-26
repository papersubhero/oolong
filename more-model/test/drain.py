"""Payload runs out. Sender and receiver must stay on the same keystream."""
import random
import sys
from util import dists
from stego.reg import get, names

PAYLOAD = "10110100" * 4


def check(name, n=200, vocab=64, key=4242):
    c = get(name)
    if not c.carries:
        return True, True
    D = dists(n, vocab, 7)
    snap = random.Random(key).getstate()

    e = random.Random()
    e.setstate(snap)
    st, bits, toks = c.new_state(), PAYLOAD, []
    for p, i in D:                       # no early break
        tid, nb, st = c.encode_step(p, i, bits, st, e)
        toks.append(tid)
        bits = bits[nb:]

    d = random.Random()
    d.setstate(snap)
    st2, out = c.new_state(), ""
    for (p, i), tid in zip(D, toks):
        b, st2 = c.decode_step(p, i, tid, st2, d)
        out += b

    return out.startswith(PAYLOAD), e.getstate() == d.getstate()


def main():
    good = True
    print("{:9s} {:>8s} {:>7s}".format("codec", "payload", "sync"))
    for n in names():
        pay, sync = check(n)
        good &= pay and sync
        print("{:9s} {:>8s} {:>7s}  {}".format(n, str(pay), str(sync),
                                               "ok" if pay and sync else "DESYNC"))
    sys.exit(0 if good else 1)


main()
