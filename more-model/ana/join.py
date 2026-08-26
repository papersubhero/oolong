#!/usr/bin/env python3
"""Concatenate shard outputs."""
import argparse
import glob
import pandas as pd


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--glob", default="res/runs/*.csv")
    a.add_argument("--out", default="res/all.csv")
    n = a.parse_args()
    files = sorted(glob.glob(n.glob))
    if not files:
        print("nothing at " + n.glob)
        return
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.to_csv(n.out, index=False)
    print("{} runs from {} files -> {}".format(len(df), len(files), n.out))
    print(df.groupby(["EXP", "ALGORITHM"]).size().to_string())


if __name__ == "__main__":
    main()
