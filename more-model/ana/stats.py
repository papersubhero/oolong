#!/usr/bin/env python3
"""Headline metrics. Decoding and conversational quality stay separate.

    python ana/stats.py --csv res/all.csv --judge gpt-4o
"""
import argparse
import pandas as pd

FAIL_NA = [None, "", "none", "None", "ERR", "nan"]


def hit(s):
    return ~s.astype(str).isin(FAIL_NA) & s.notna()


def truthy(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True)
    a.add_argument("--judge", default="gpt-4o")
    a.add_argument("--by", nargs="+", default=["ALGORITHM", "LANGUAGE", "MODEL", "TEMPERATURE"])
    a.add_argument("--out", default=None)
    n = a.parse_args()

    tag = n.judge.replace("/", "-")
    df = pd.read_csv(n.csv, low_memory=False)

    for k in ("format", "role", "collapse"):
        c = "J_{}_{}".format(tag, k)
        df["_" + k] = hit(df[c]) if c in df else False
    df["_bad"] = df["_format"] | df["_role"] | df["_collapse"]
    df["_clean"] = ~df["_bad"]

    df["_dec"] = truthy(df["OK_12"]) & truthy(df["OK_21"])
    df["_both"] = df["_dec"] & df["_clean"]
    df["_carry"] = df["ALGORITHM"].str.lower() != "vanilla"

    msg = [c for c in df.columns if c.startswith(("ANSWER_", "MESSAGE_"))]
    df["_empty"] = df[msg].isna().any(axis=1) | (df[msg].astype(str)
                                                 .apply(lambda s: s.str.strip() == "").any(axis=1))

    bits = [c for c in df.columns if c.startswith("BITS_")]
    ents = [c for c in df.columns if c.startswith("H_")]
    df["_bits"] = df[bits].apply(pd.to_numeric, errors="coerce").sum(axis=1) if bits else 0.0
    df["_H"] = df[ents].apply(pd.to_numeric, errors="coerce").sum(axis=1) if ents else 0.0
    df["_util"] = (df["_bits"] / df["_H"]).where(df["_H"] > 0)

    nat = "J_{}_natural".format(tag)
    df["_nat"] = pd.to_numeric(df[nat], errors="coerce") if nat in df else float("nan")

    def block(g):
        c = g[g["_carry"]]
        return pd.Series({
            "n": len(g),
            "clean%": round(100 * g["_clean"].mean(), 1),
            "role%": round(100 * g["_role"].mean(), 1),
            "collapse%": round(100 * g["_collapse"].mean(), 1),
            "format%": round(100 * g["_format"].mean(), 1),
            "empty%": round(100 * g["_empty"].mean(), 1),
            "natural": round(g["_nat"].mean(), 2),
            "decode%": round(100 * c["_dec"].mean(), 1) if len(c) else None,
            "both%": round(100 * c["_both"].mean(), 1) if len(c) else None,
            "util": round(c["_util"].mean(), 3) if len(c) else None,
            "secs": round(pd.to_numeric(g.get("SECS"), errors="coerce").mean(), 1),
        })

    first = None
    for key in n.by:
        if key not in df:
            continue
        t = df.groupby(key).apply(block, include_groups=False)
        print("\n=== by {} ===".format(key))
        print(t.to_string())
        first = t if first is None else first

    if n.out and first is not None:
        first.to_csv(n.out)
        print("\nwrote " + n.out)


if __name__ == "__main__":
    main()
