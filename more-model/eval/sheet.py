#!/usr/bin/env python3
"""Split the subset into one blank annotation sheet per language."""
import argparse
import csv
import os

COLS = ["ROW", "LANGUAGE", "MODEL", "ALGORITHM", "TEMPLATE", "CONVERSATION",
        "H_FORMAT", "H_ROLE", "H_COLLAPSE", "H_NATURAL", "H_NOTES"]
HELP = ("H_FORMAT/H_ROLE/H_COLLAPSE: first message number, or blank if absent. "
        "H_NATURAL: 1-5. Do not look at the judge columns.")


def convo(row):
    a1 = row.get("ROLE_1") or "Agent 1"
    a2 = row.get("ROLE_2") or "Agent 2"
    out, i = [], 1
    for t in range(1, int(row.get("TURNS") or 0) + 1):
        for col, who in (("ANSWER_%d" % t, a2), ("MESSAGE_%d" % t, a1)):
            txt = (row.get(col) or "").strip()
            if txt:
                out.append("[{}] {}: {}".format(i, who, txt))
                i += 1
    return "\n".join(out)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True)
    a.add_argument("--out", default="res/annot")
    n = a.parse_args()

    os.makedirs(n.out, exist_ok=True)
    rows = list(csv.DictReader(open(n.csv, encoding="utf-8")))
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r.get("LANGUAGE", "?"), []).append((i, r))

    for lang, items in by.items():
        path = os.path.join(n.out, "annot_{}.csv".format(lang.lower()))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for i, r in items:
                w.writerow({"ROW": i, "LANGUAGE": lang, "MODEL": r.get("MODEL"),
                            "ALGORITHM": r.get("ALGORITHM"), "TEMPLATE": r.get("TEMPLATE"),
                            "CONVERSATION": convo(r)})
        print("{:9s} {:4d} -> {}".format(lang, len(items), path))
    print(HELP)


if __name__ == "__main__":
    main()
