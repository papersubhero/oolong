#!/usr/bin/env python3
"""Bilateral self-play. Two agents alternate; each hides bits in its own replies.

    python run/main.py --cfg cfg/out/e1__Discop__Qwen-Qwen3.5-9B.csv
"""
import argparse
import csv
import gc
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cfg"))
sys.path.insert(0, HERE)

import torch                               # noqa: E402
from stego.reg import get                  # noqa: E402
import text                                # noqa: E402
import gen                                 # noqa: E402

U1, U2 = "<|user1|>", "<|user2|>"


def to_bits(s):
    return "".join(format(b, "08b") for b in s.encode("utf-8")) if s else ""


def one(row, model, tok):
    cfg = {"MODEL": row["MODEL"], "TEMPERATURE": float(row["TEMPERATURE"]),
           "TOP_P": float(row["TOP_P"]), "GEN_LEN": int(row["GEN_LEN"])}
    lang, tpl = row["LANGUAGE"], row["TEMPLATE"]
    style = "chat" if row.get("STYLE", "chat") == "chat" else "raw"
    by_text = row.get("CHANNEL", "ids") == "text"
    codec = get(row["ALGORITHM"])
    turns_n = int(row["TURNS"])

    r1, r2 = text.role_names(tpl, lang)
    roles = [r1, r2]
    turns = text.seed_turns(tpl, lang)

    pay = {U1: "", U2: ""}
    if codec.carries:
        pay[U1] = to_bits(row.get("PAY_1", ""))
        pay[U2] = to_bits(row.get("PAY_2", ""))
    left = dict(pay)
    got = {U1: "", U2: ""}
    enc = {m: codec.new_state() for m in pay}
    dec = {m: codec.new_state() for m in pay}

    rng = random.Random(int(row["SEED"]))
    out, order = {}, [(U2, "ANSWER"), (U1, "MESSAGE")]

    for t in range(1, turns_n + 1):
        for marker, tag in order:
            ids = gen.context(tok, cfg["MODEL"], tpl, lang, marker, turns, style, cfg["GEN_LEN"])
            snap = rng.getstate()
            res = gen.write(model, tok, codec, ids, cfg, rng, enc[marker], left[marker], roles)
            enc[marker] = res["state"]
            left[marker] = left[marker][res["bits"]:]

            body = gen.clean(res["raw"], roles)
            fb = 0
            if not body:
                body, fb = row.get("FALLBACK") or "...", 1

            if codec.carries:
                rng.setstate(snap)
                carrier = tok(body, add_special_tokens=False).input_ids if by_text else res["toks"]
                bits, dec[marker] = gen.read(model, tok, codec, ids, carrier, cfg, rng, dec[marker])
                got[marker] += bits

            turns.append((marker, body))
            out["{}_{}".format(tag, t)] = body
            out["BITS_{}_{}".format(tag, t)] = res["bits"]
            out["H_{}_{}".format(tag, t)] = round(res["entropy"], 3)
            out["FB_{}_{}".format(tag, t)] = fb

    if codec.carries:
        ok12 = bool(pay[U1]) and got[U1].startswith(pay[U1])
        ok21 = bool(pay[U2]) and got[U2].startswith(pay[U2])
    else:
        ok12 = ok21 = ""

    head = {k: row[k] for k in ("EXP", "LANGUAGE", "MODEL", "ALGORITHM", "TEMPERATURE",
                                "TOP_P", "GEN_LEN", "SEED", "TURNS", "TEMPLATE",
                                "STYLE", "CHANNEL", "PAY_1", "PAY_2", "PAY_IDX")}
    head.update({"ROLE_1": r1, "ROLE_2": r2, "OK_12": ok12, "OK_21": ok21})
    head.update(out)
    return head


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cfg", required=True)
    a.add_argument("--out", default=None)
    a.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    n = a.parse_args()

    rows = list(csv.DictReader(open(n.cfg, newline="", encoding="utf-8")))
    if not rows:
        print("empty config")
        return

    stem = os.path.splitext(os.path.basename(n.cfg))[0]
    path = n.out or os.path.join(ROOT, "res", "runs", stem + ".csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    name = rows[0]["MODEL"]
    print("[load] {} ({} runs)".format(name, len(rows)))
    model, tok = gen.load(name, n.token)

    done = 0
    if os.path.exists(path):
        done = sum(1 for _ in open(path, encoding="utf-8")) - 1
        print("[resume] {} rows already there".format(done))

    for i, row in enumerate(rows):
        if i < done:
            continue
        t0, err = time.time(), ""
        try:
            rec = one(row, model, tok)
        except Exception as exc:
            err = "{}: {}".format(type(exc).__name__, exc)
            rec = {k: row.get(k, "") for k in ("EXP", "LANGUAGE", "MODEL", "ALGORITHM",
                                               "TEMPERATURE", "TEMPLATE")}
            print("  [err] row {}: {}".format(i, err))
        rec["SECS"] = round(time.time() - t0, 2)
        rec["ERR"] = err
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rec.keys()), extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(rec)
        if (i + 1) % 10 == 0:
            print("  {}/{}".format(i + 1, len(rows)))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[done] -> {}".format(path))


if __name__ == "__main__":
    main()
