"""Vanilla sampling. Control arm, carries nothing."""
from .base import Codec


class Vanilla(Codec):
    name = "vanilla"
    carries = False

    def encode_step(self, probs, indices, bits, state, rng):
        r, acc = rng.random(), 0.0
        for p, tid in zip(probs, indices):
            acc += p
            if r <= acc:
                return tid, 0, state
        return indices[-1], 0, state

    def decode_step(self, probs, indices, token_id, state, rng):
        rng.random()          # keep draws symmetric
        return "", state
