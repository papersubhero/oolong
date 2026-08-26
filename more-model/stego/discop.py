"""Discop (S&P 2023). Huffman tree, two distribution copies offset by 1/2."""
from collections import deque
from .base import Codec


class Node:
    __slots__ = ("prob", "left", "right", "index", "path")

    def __init__(self, prob, left=None, right=None, index=-1, path=9):
        self.prob, self.left, self.right = prob, left, right
        self.index, self.path = index, path


def build(indices, probs, target=-1):
    q1 = deque(
        Node(probs[i], None, None, indices[i], 0 if target == indices[i] else 9)
        for i in range(len(indices) - 1, -1, -1)
    )
    q2 = deque()
    while len(q1) + len(q2) > 1:
        pair = []
        for _ in range(2):
            if q1 and q2:
                pair.append(q1.popleft() if q1[0].prob < q2[0].prob else q2.popleft())
            else:
                pair.append(q1.popleft() if q1 else q2.popleft())
        a, b = pair
        path = -1 if a.path != 9 else (1 if b.path != 9 else 9)
        q2.append(Node(a.prob + b.prob, a, b, -1, path))
    return q2[0] if q2 else q1[0]


class Discop(Codec):
    name = "Discop"

    def encode_step(self, probs, indices, bits, state, rng):
        node = build(indices, probs)
        depth = used = 0
        while node.index == -1:
            ptr = rng.random()
            p0, p1 = ptr * node.prob, (ptr + 0.5) * node.prob
            if p1 > node.prob:
                p1 -= node.prob
            a = -1 if p0 < node.left.prob else 1
            b = -1 if p1 < node.left.prob else 1
            take = a if depth >= len(bits) else (b if bits[depth] == "1" else a)
            node = node.right if take == 1 else node.left
            if a != b:
                if depth < len(bits):
                    used += 1
                depth += 1
        return node.index, used, state

    def decode_step(self, probs, indices, token_id, state, rng):
        node = build(indices, probs, target=token_id)
        out = []
        while node.index == -1:
            ptr = rng.random()
            p0, p1 = ptr * node.prob, (ptr + 0.5) * node.prob
            if p1 > node.prob:
                p1 -= node.prob
            a = -1 if p0 < node.left.prob else 1
            b = -1 if p1 < node.left.prob else 1
            if a != b:
                if a == -1:
                    out.append("0" if node.path == -1 else "1")
                else:
                    out.append("1" if node.path == -1 else "0")
                node = node.left if node.path == -1 else node.right
            else:
                node = node.left if a == -1 else node.right
        return "".join(out), state
