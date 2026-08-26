"""Per-token codec interface."""
from abc import ABC, abstractmethod


class Codec(ABC):
    """probs/indices are sorted desc, float64, already temp+top_p filtered.

    encode_step -> (token_id, n_bits, state)   n_bits = payload bits consumed
    decode_step -> (bits, state)

    rng is the shared keystream. Sender and receiver must draw from it
    identically, including after the payload runs out.
    """

    name = "base"
    stateful = False
    carries = True

    def new_state(self):
        return None

    @abstractmethod
    def encode_step(self, probs, indices, bits, state, rng):
        ...

    @abstractmethod
    def decode_step(self, probs, indices, token_id, state, rng):
        ...
