"""Logit range check. Admission test for any model before a sweep.

Raw logit scale is arbitrary per model, so a fixed clamp silently flattens the
distribution on some of them. Run this before adding a model.
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stego import probs as P              # noqa: E402

PROBE = ("Anna: Good morning, are you the new supervisor?\n"
         "Mark: Yes, I am. Nice to meet you.\nAnna:")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--temp", type=float, default=0.7)
    a.add_argument("--top-p", type=float, default=0.92)
    n = a.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("{:38s} {:>20s} {:>8s} {:>8s}".format("model", "logit range", "nucleus", "H bits"))
    for name in n.models:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            m = AutoModelForCausalLM.from_pretrained(name)
            m.eval()
            with torch.no_grad():
                lg = m(tok(PROBE, return_tensors="pt").input_ids).logits[0, -1, :].float().numpy()
            p, _ = P.get(lg.astype(np.float64), n.temp, n.top_p)
            print("{:38s} {:>20s} {:8d} {:8.2f}".format(
                name, "[{:.0f}, {:.0f}]".format(lg.min(), lg.max()), len(p), P.entropy(p)))
            del m
        except Exception as exc:
            print("{:38s} {}: {}".format(name, type(exc).__name__, str(exc)[:40]))
    sys.exit(0)


main()
