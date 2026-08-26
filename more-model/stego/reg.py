"""Name -> codec."""
from .van import Vanilla
from .discop import Discop
from .meteor import Meteor
# from .spar import SparSamp
# from .shim import Shimmer
# from .imec import IMEC

TABLE = {c.name.lower(): c for c in (Vanilla, Discop, Meteor, SparSamp, Shimmer, IMEC)}


def get(name):
    key = str(name).lower()
    if key not in TABLE:
        raise KeyError("unknown algorithm {!r}; have {}".format(name, sorted(TABLE)))
    return TABLE[key]()


def names():
    return sorted(TABLE)
