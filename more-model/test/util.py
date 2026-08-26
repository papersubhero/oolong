"""Synthetic distributions."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def dists(n, vocab, seed, alpha=2.0):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        raw = [r.random() ** alpha + 1e-6 for _ in range(vocab)]
        s = sum(raw)
        idx = list(range(vocab))
        r.shuffle(idx)
        out.append((sorted((x / s for x in raw), reverse=True), idx))
    return out


def one(vocab, seed, alpha=1.5):
    r = random.Random(seed)
    raw = [r.random() ** alpha + 0.02 for _ in range(vocab)]
    s = sum(raw)
    return sorted((x / s for x in raw), reverse=True), list(range(vocab))
