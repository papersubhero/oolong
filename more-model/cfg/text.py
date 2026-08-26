"""Prompt construction. Single source of truth for sender and receiver."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}

# Chat unless listed here or suffixed -base. Modern instruct checkpoints
# often carry no suffix at all (Qwen3.5-9B is instruct, -9B-Base is not).
BASE = {"gpt2", "allenai/Olmo-3-1025-7B", "swiss-ai/Apertus-8B-2509"}


def load(name):
    if name not in _CACHE:
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            _CACHE[name] = json.load(f)
    return _CACHE[name]


def is_chat(model):
    return model not in BASE and not model.lower().endswith("-base")


def parse(seed):
    parts = re.split(r"(<\|user[12]\|>)", seed)
    turns, i = [], 1
    while i < len(parts):
        txt = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if txt:
            turns.append((parts[i].strip(), txt))
        i += 2
    return turns


def role(marker, kind, lang, roles):
    return roles["types"][kind]["role1" if marker == "<|user1|>" else "role2"][lang]


def build(model, template, lang, marker, turns, style="chat"):
    roles = load("roles.json")
    rules = load("rules.json")[lang]
    kind = roles["template_type"][template]

    r1 = roles["types"][kind]["role1"][lang]
    r2 = roles["types"][kind]["role2"][lang]
    who = role(marker, kind, lang, roles)
    persona = roles["types"][kind]["persona1" if marker == "<|user1|>" else "persona2"][lang]
    intro = rules["base_intro"].format(role1=r1, role2=r2)
    script = "\n".join("{}: {}".format(role(m, kind, lang, roles), t) for m, t in turns)

    if style == "chat" and is_chat(model):
        return {"mode": "chat", "who": who,
                "system": "{}\n\n{}".format(persona, rules["system_rules"]),
                "user": "{}\n\n{}".format(intro, script)}
    return {"mode": "base", "who": who,
            "text": "{} {}\n\n{}\n{}:".format(persona, intro, script, who)}


def role_names(template, lang):
    roles = load("roles.json")
    kind = roles["template_type"][template]
    return roles["types"][kind]["role1"][lang], roles["types"][kind]["role2"][lang]


def seed_turns(template, lang):
    return parse(load("seeds.json")["seeds"][lang][template])


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--template", default="S2")
    a.add_argument("--lang", default="English")
    a.add_argument("--model", default="Qwen/Qwen3.5-9B")
    a.add_argument("--marker", default="<|user1|>")
    n = a.parse_args()
    s = build(n.model, n.template, n.lang, n.marker, seed_turns(n.template, n.lang))
    print(s.get("text") or "[SYSTEM]\n{}\n\n[USER]\n{}".format(s["system"], s["user"]))
