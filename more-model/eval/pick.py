#!/usr/bin/env python3
"""Stratified subset for human annotation. Balanced over language, model,
algorithm and outcome so judge bias can be measured per language."""
import argparse
import csv
import random


def outcome(row, judge):
    tag = judge.replace("/", "-")
    for suf in ("format", "role", "collapse"):
        v = row.get("J_{}_{}".format(tag, suf))
        if v not in (None, "", "none", "None", "ERR"):
            return "fail"
    return "clean"


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True)
    a.add_argument("--n", type=int, default=400)
    a.add_argument("--judge", default="gpt-4o")
    a.add_argument("--by", nargs="+", default=["LANGUAGE", "MODEL", "ALGORITHM", "outcome"])
    a.add_argument("--out", default="res/subset.csv")
    a.add_argument("--seed", type=int, default=42)
    n = a.parse_args()

    rows = list(csv.DictReader(open(n.csv, encoding="utf-8")))
    rng = random.Random(n.seed)
    bins = {}
    for r in rows:
        r["outcome"] = outcome(r, n.judge)
        bins.setdefault(tuple(r.get(k, "") for k in n.by), []).append(r)
    for b in bins.values():
        rng.shuffle(b)

    keys, out, i = list(bins.keys()), [], 0
    while len(out) < n.n and any(bins.values()):
        b = bins[keys[i % len(keys)]]
        if b:
            out.append(b.pop())
        i += 1
        if i > n.n * 10:
            break

    fields = list(rows[0].keys()) + (["outcome"] if "outcome" not in rows[0] else [])
    with open(n.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print("{} convos, {} strata -> {}".format(len(out), len(bins), n.out))


if __name__ == "__main__":
    main()
