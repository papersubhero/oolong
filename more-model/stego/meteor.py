"""Meteor (CCS 2021). Arithmetic coding with an XOR-masked keystream."""
from .base import Codec

PREC = 32
FULL = 1 << PREC


def cdf(probs):
    """Integer CDF tiling [0, 2^PREC). Quantise the running CDF, not each prob:
    that keeps it monotone and makes the last bound exactly FULL."""
    tot = sum(probs)
    out, acc, prev = [], 0.0, 0
    for p in probs:
        acc += p
        c = int(acc / tot * FULL)
        if c < prev:
            c = prev
        out.append(c)
        prev = c
    out[-1] = FULL
    return out


def pick(cum, r):
    for j, c in enumerate(cum):
        if c > r:
            return j
    return len(cum) - 1


def shared(a, b):
    x = a ^ b
    return PREC if x == 0 else PREC - x.bit_length()


class Meteor(Codec):
    name = "Meteor"

    def encode_step(self, probs, indices, bits, state, rng):
        cum = cdf(probs)
        mask = rng.getrandbits(PREC)

        k = min(len(bits), PREC)
        m = 0
        for j in range(k):
            if bits[j] == "1":
                m |= 1 << (PREC - 1 - j)

        sel = pick(cum, m ^ mask)
        lo = cum[sel - 1] if sel > 0 else 0
        return indices[sel], min(shared(lo, cum[sel] - 1), k), state

    def decode_step(self, probs, indices, token_id, state, rng):
        cum = cdf(probs)
        mask = rng.getrandbits(PREC)

        sel = min(indices.index(token_id), len(cum) - 1)
        lo = cum[sel - 1] if sel > 0 else 0
        hi = cum[sel] - 1
        n = shared(lo, hi)

        out = []
        for b in range(n):
            sh = PREC - 1 - b
            out.append("1" if ((hi >> sh) & 1) ^ ((mask >> sh) & 1) else "0")
        return "".join(out), state
