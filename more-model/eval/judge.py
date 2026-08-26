#!/usr/bin/env python3
"""LLM judge. Four labels per conversation, one column set per judge.

    python eval/judge.py --csv res/all.csv --judges gpt-4o
    python eval/judge.py --csv res/subset.csv --judges gpt-4o claude-sonnet-4-5 --n 5
"""
import argparse
import csv
import json
import os
import statistics

PROMPT = """Analyze the following numbered conversation between two people, in {lang}.
Agent 1 = {a1} ; Agent 2 = {a2}.

Report FOUR things. For 1-3 give the FIRST message number where it happens, else null.

1 FORMAT: artifacts that never appear in a human chat. Control tokens
  (<|im_end|>, <|endoftext|>, </s>, <pad>), leaked role labels ("{a1}:", "assistant:"),
  raw HTML/XML, or <think> reasoning leaking into the reply.

2 ROLE: an agent breaks character. States or implies it is an AI, drops its persona,
  contradicts the scenario, or replies in a language other than {lang}.

3 COLLAPSE: degeneration. Near-exact repeat of an earlier message, one word or character
  repeated, or a loop across turns. Laughter (haha, jaja) and emphasis (noo, siii) are not collapse.

4 NATURALNESS 1-5: could two real humans have written this whole exchange?
  5 fully natural, 3 somewhat robotic, 1 clearly machine. Judge the whole conversation.

Conversation:
{convo}

Reply with JSON only:
{{"format": <int|null>, "format_why": "<str>",
  "role": <int|null>, "role_why": "<str>",
  "collapse": <int|null>, "collapse_why": "<str>",
  "natural": <int 1-5>, "natural_why": "<str>"}}"""


def ask(model, prompt, temp=0.2):
    if model.startswith(("gpt", "o1", "o3", "o4")):
        from openai import OpenAI
        c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        r = c.chat.completions.create(
            model=model, temperature=temp, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "Conversation evaluator. Reply valid JSON."},
                      {"role": "user", "content": prompt}])
        return json.loads(r.choices[0].message.content)
    if model.startswith("claude"):
        import anthropic
        c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        r = c.messages.create(model=model, max_tokens=1024, temperature=temp,
                              messages=[{"role": "user", "content": prompt + "\n\nJSON only."}])
        t = r.content[0].text
        return json.loads(t[t.find("{"):t.rfind("}") + 1])
    from openai import OpenAI
    c = OpenAI(api_key=os.environ.get("JUDGE_KEY", "x"),
               base_url=os.environ.get("JUDGE_URL", "http://localhost:8000/v1"))
    r = c.chat.completions.create(model=model, temperature=temp,
                                  messages=[{"role": "user", "content": prompt}])
    t = r.choices[0].message.content
    return json.loads(t[t.find("{"):t.rfind("}") + 1])


def convo(row):
    a1 = row.get("ROLE_1") or "Agent 1"
    a2 = row.get("ROLE_2") or "Agent 2"
    lines, i = [], 1
    for t in range(1, int(row.get("TURNS") or 0) + 1):
        for col, who in (("ANSWER_%d" % t, a2), ("MESSAGE_%d" % t, a1)):
            txt = (row.get(col) or "").strip()
            if txt:
                lines.append("[{}] {}: {}".format(i, who, txt))
                i += 1
    return "\n".join(lines), a1, a2


def vote(vals):
    """Median of the non-null indices, null if most judges said null."""
    hit = [v for v in vals if v is not None]
    return int(statistics.median(hit)) if len(hit) > len(vals) / 2 else None


def label(row, judges, n):
    body, a1, a2 = convo(row)
    if not body:
        return row
    prompt = PROMPT.format(lang=row.get("LANGUAGE", "English"), a1=a1, a2=a2, convo=body)
    for j in judges:
        acc = {"format": [], "role": [], "collapse": [], "natural": [], "why": []}
        for _ in range(n):
            try:
                r = ask(j, prompt)
            except Exception as exc:
                acc["why"].append("ERR:" + type(exc).__name__)
                continue
            for k in ("format", "role", "collapse"):
                acc[k].append(r.get(k))
            if isinstance(r.get("natural"), (int, float)):
                acc["natural"].append(float(r["natural"]))
            acc["why"].append(json.dumps({k: r.get(k) for k in r if k.endswith("_why")},
                                         ensure_ascii=False))
        tag = j.replace("/", "-")
        for k in ("format", "role", "collapse"):
            row["J_{}_{}".format(tag, k)] = vote(acc[k]) if acc[k] else "ERR"
        row["J_{}_natural".format(tag)] = round(statistics.mean(acc["natural"]), 2) if acc["natural"] else "ERR"
        row["J_{}_why".format(tag)] = " || ".join(acc["why"])[:2000]
    return row


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True)
    a.add_argument("--judges", nargs="+", default=["gpt-4o"])
    a.add_argument("--n", type=int, default=1)
    a.add_argument("--every", type=int, default=25)
    n = a.parse_args()

    rows = list(csv.DictReader(open(n.csv, encoding="utf-8")))
    fields = list(rows[0].keys()) if rows else []
    for j in n.judges:
        tag = j.replace("/", "-")
        for suf in ("format", "role", "collapse", "natural", "why"):
            col = "J_{}_{}".format(tag, suf)
            if col not in fields:
                fields.append(col)

    def flush():
        tmp = n.csv + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, n.csv)

    keys = ["J_{}_format".format(j.replace("/", "-")) for j in n.judges]
    done = 0
    for i, row in enumerate(rows):
        if all(str(row.get(k, "")).strip() for k in keys):
            continue
        try:
            label(row, n.judges, n.n)
            done += 1
        except Exception as exc:
            print("[err] row {}: {}".format(i, exc))
        if done and done % n.every == 0:
            flush()
            print("  {}/{}".format(i + 1, len(rows)))
    flush()
    print("[done] {} rows by {}".format(done, n.judges))


if __name__ == "__main__":
    main()
