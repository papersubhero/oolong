#!/usr/bin/env python3
"""Vector figures. PDF, not PNG."""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                       # noqa: E402

NA = [None, "", "none", "None", "ERR", "nan"]


def hit(s):
    return ~s.astype(str).isin(NA) & s.notna()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True)
    a.add_argument("--judge", default="gpt-4o")
    a.add_argument("--out", default="res/figs")
    n = a.parse_args()

    tag = n.judge.replace("/", "-")
    os.makedirs(n.out, exist_ok=True)
    df = pd.read_csv(n.csv, low_memory=False)
    for k in ("format", "role", "collapse"):
        c = "J_{}_{}".format(tag, k)
        df["_" + k] = hit(df[c]) if c in df else False
    df["_clean"] = ~(df["_format"] | df["_role"] | df["_collapse"])
    nat = "J_{}_natural".format(tag)
    df["_nat"] = pd.to_numeric(df[nat], errors="coerce") if nat in df else float("nan")

    # clean rate by algorithm, control highlighted
    g = df.groupby("ALGORITHM")["_clean"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(g.index, g.values,
            color=["#333" if i.lower() == "vanilla" else "#999" for i in g.index])
    ax.set_xlabel("clean conversations (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(n.out, "clean.pdf"))

    # naturalness by algorithm
    fig, ax = plt.subplots(figsize=(6, 3.2))
    order = sorted(df["ALGORITHM"].dropna().unique())
    ax.boxplot([df.loc[df.ALGORITHM == k, "_nat"].dropna() for k in order], labels=order)
    ax.set_ylabel("naturalness (1-5)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(n.out, "natural.pdf"))

    # failure by language
    fig, ax = plt.subplots(figsize=(6, 3.2))
    t = df.groupby("LANGUAGE")[["_format", "_role", "_collapse"]].mean().mul(100)
    t.plot(kind="bar", ax=ax)
    ax.set_ylabel("failure rate (%)")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(n.out, "language.pdf"))
    print("wrote 3 pdfs -> " + n.out)


if __name__ == "__main__":
    main()
