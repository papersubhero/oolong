"""Round trip: encode then decode recovers the payload."""
import random
import sys
from util import dists
from stego.reg import get, names

PAYLOADS = {
    "mixed": "".join(random.Random(1).choice("01") for _ in range(64)),
    "zeros": "0" * 64,
    "ones": "1" * 64,
    "alt": "01" * 32,
}


def trip(name, payload, n=600, vocab=64, seed=7, key=12345):
    c = get(name)
    if not c.carries:
        return True, "control"
    D = dists(n, vocab, seed)
    snap = random.Random(key).getstate()

    e = random.Random()
    e.setstate(snap)
    st, bits, toks, used = c.new_state(), payload, [], 0
    for p, i in D:
        tid, nb, st = c.encode_step(p, i, bits, st, e)
        toks.append(tid)
        used += min(nb, len(bits))
        bits = bits[nb:]
        if not bits:
            break

    d = random.Random()
    d.setstate(snap)
    st2, out = c.new_state(), ""
    for (p, i), tid in zip(D, toks):
        b, st2 = c.decode_step(p, i, tid, st2, d)
        out += b

    ok = out.startswith(payload) and used >= len(payload)
    return ok, "{}b in {} tokens".format(len(payload), len(toks))


def main():
    good = True
    for n in names():
        for label, pay in PAYLOADS.items():
            try:
                ok, msg = trip(n, pay)
            except Exception as exc:
                ok, msg = False, "{}: {}".format(type(exc).__name__, exc)
            print("[{}] {:9s} {:6s} {}".format("ok " if ok else "FAIL", n, label, msg))
            good &= ok
    sys.exit(0 if good else 1)


main()
