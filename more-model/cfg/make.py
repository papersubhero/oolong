#!/usr/bin/env python3
"""Build run configs. One CSV per shard; the runner names its output after the
shard file, so parallel array tasks never share a path."""
import argparse
import csv
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

CORE = [
    "Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B",
    "swiss-ai/Apertus-v1.5-8B", "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-4-12B-it", "openai/gpt-oss-20b",
]
LADDER = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"]
POST = ["allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-SFT",
        "allenai/Olmo-3-7B-Instruct-DPO", "allenai/Olmo-3-7B-Instruct"]
ARCH = ["Qwen/Qwen3.5-35B-A3B", "openai/gpt-oss-20b", "swiss-ai/Apertus-v1.5-70B"]

ALL = ["vanilla", "Discop", "Meteor", "iMEC", "SparSamp", "Shimmer"]
PAIR = ["vanilla", "Discop"]
LANGS = ["English", "Spanish", "Russian", "German", "Chinese"]
SHORT = ["S1", "S4", "S7"]
LONG = ["S2", "S3", "S5", "S6", "S8", "S9", "S10"]
CODE = {"English": "en", "Spanish": "es", "Russian": "ru", "German": "de", "Chinese": "zh"}

# algos, models, temps, langs, templates, top_p
EXP = {
    "e1": (ALL, CORE, [0.3, 0.5, 0.7], LANGS, SHORT, 0.92),
    "e2": (PAIR, CORE, [0.7], LANGS, LONG, 0.92),
    "e3": (PAIR, LADDER, [0.7], LANGS, SHORT, 0.92),
    "e4": (PAIR, POST, [0.7], LANGS, SHORT, 0.92),
    "e5": (PAIR, ARCH, [0.7], LANGS, SHORT, 0.92),
    "e6": (["vanilla", "Discop", "Meteor"], CORE[:4], [0.7], LANGS, SHORT, 1.0),
    "pilot": (ALL, CORE[2:4], [0.7], ["English", "Russian"], ["S1", "S7"], 0.92),
}

FIELDS = ["EXP", "LANGUAGE", "MODEL", "ALGORITHM", "TEMPERATURE", "TOP_P", "GEN_LEN",
          "SEED", "TURNS", "TEMPLATE", "STYLE", "CHANNEL", "PAY_1", "PAY_2", "PAY_IDX", "FALLBACK"]
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--exp", nargs="+", default=["e1"], choices=sorted(EXP))
    a.add_argument("--payloads", type=int, default=1)
    a.add_argument("--style", default="chat", choices=["chat", "raw"])
    a.add_argument("--channel", default="ids", choices=["ids", "text"])
    a.add_argument("--turns", type=int, default=20)
    a.add_argument("--gen-len", type=int, default=200)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", default=os.path.join(HERE, "out"))
    n = a.parse_args()

    rules = json.load(open(os.path.join(HERE, "rules.json"), encoding="utf-8"))
    rng = random.Random(n.seed)
    rows = []
    for exp in n.exp:
        algos, models, temps, langs, templates, top_p = EXP[exp]
        for algo in algos:
            carries = algo != "vanilla"
            for model in models:
                for lang in langs:
                    for tpl in templates:
                        for temp in temps:
                            for k in range(n.payloads):
                                p1 = "".join(rng.choice(ALPHABET) for _ in range(4)) if carries else ""
                                p2 = "".join(rng.choice(ALPHABET) for _ in range(3)) if carries else ""
                                rows.append({
                                    "EXP": exp, "LANGUAGE": lang, "MODEL": model,
                                    "ALGORITHM": algo, "TEMPERATURE": temp, "TOP_P": top_p,
                                    "GEN_LEN": n.gen_len, "SEED": n.seed, "TURNS": n.turns,
                                    "TEMPLATE": tpl, "STYLE": n.style, "CHANNEL": n.channel,
                                    "PAY_1": p1, "PAY_2": p2, "PAY_IDX": k,
                                    "FALLBACK": rules[lang]["fallback"],
                                })

    os.makedirs(n.out, exist_ok=True)
    shards = {}
    for r in rows:
        key = "{}__{}__{}".format(r["EXP"], r["ALGORITHM"], r["MODEL"].replace("/", "-"))
        shards.setdefault(key, []).append(r)
    for key, srows in shards.items():
        with open(os.path.join(n.out, key + ".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(srows)

    json.dump({"runs": len(rows), "shards": len(shards), "exp": n.exp,
               "payloads": n.payloads, "style": n.style, "channel": n.channel},
              open(os.path.join(n.out, "manifest.json"), "w"), indent=2)
    print("{} runs, {} shards -> {}".format(len(rows), len(shards), n.out))


if __name__ == "__main__":
    main()
