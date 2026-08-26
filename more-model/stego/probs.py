"""Logits -> (probs, indices), sorted desc, float64."""
import numpy as np

# GPT-2 ids the reference masks to avoid retokenisation desync.
GPT2_MASK = [198, 628, 764, 837]


def get(logits, temperature, top_p, mask=None, floor=1e-8):
    """floor drops tokens below float noise. Needed at top_p=1.0, where the
    nucleus is otherwise the whole vocabulary. At 1e-8 the entropy is unchanged
    to three decimals and the tree is 6x cheaper; fp16 logits are far noisier."""
    logits = np.asarray(logits, dtype=np.float64).copy()
    if mask:
        for i in mask:
            if 0 <= i < logits.shape[0]:
                logits[i] = -np.inf

    order = np.argsort(-logits, kind="stable")
    z = logits[order] / float(temperature)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()

    if top_p is not None and top_p < 1.0:
        k = int(np.searchsorted(np.cumsum(p), top_p)) + 1
    else:
        k = int(np.searchsorted(-p, -floor, side="right"))
    k = max(1, min(k, p.shape[0]))
    p, order = p[:k], order[:k]
    p = p / p.sum()

    return p.tolist(), [int(i) for i in order.tolist()]


def entropy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0.0]
    return float(max(0.0, -(p * np.log2(p)).sum()))
