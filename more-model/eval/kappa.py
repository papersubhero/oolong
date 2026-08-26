#!/usr/bin/env python3
"""Cohen's kappa between judges, and judge vs human."""
import argparse
import csv
import itertools


def flag(v):
    return v not in (None, "", "none", "None", "ERR", "nan")


def col(source, kind):
    return "H_" + kind.upper() if source == "H" else "J_{}_{}".format(source.replace("/", "-"), kind)


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    n = len(pairs)
    obs = sum(1 for x, y in pairs if x == y) / n
    pa = sum(1 for x, _ in pairs if x) / n
    pb = sum(1 for _, y in pairs if y) / n
    exp = pa * pb + (1 - pa) * (1 - pb)
    return (None if exp == 1 else (obs - exp) / (1 - exp)), n


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True)
    a.add_argument("--sources", nargs="+", required=True, help="judge names, H for human")
    a.add_argument("--kinds", nargs="+", default=["format", "role", "collapse"])
    n = a.parse_args()

    rows = list(csv.DictReader(open(n.csv, encoding="utf-8")))
    print("{:28s} {:10s} {:>8s} {:>6s}".format("pair", "label", "kappa", "n"))
    for s1, s2 in itertools.combinations(n.sources, 2):
        for kind in n.kinds:
            c1, c2 = col(s1, kind), col(s2, kind)
            if c1 not in rows[0] or c2 not in rows[0]:
                continue
            v1 = [flag(r.get(c1)) if r.get(c1) is not None else None for r in rows]
            v2 = [flag(r.get(c2)) if r.get(c2) is not None else None for r in rows]
            k, m = kappa(v1, v2)
            print("{:28s} {:10s} {:>8s} {:6d}".format(
                "{} vs {}".format(s1, s2)[:28], kind,
                "n/a" if k is None else "{:.3f}".format(k), m))


if __name__ == "__main__":
    main()
