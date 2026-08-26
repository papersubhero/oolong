"""Generation and decoding, one token at a time. Algorithm agnostic."""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cfg"))

from stego import probs as P               # noqa: E402
import text                                # noqa: E402

MAX_CTX = 2048
STOP = ["<|endoftext|>", "<|end_of_text|>", "<|eot_id|>", "<eos>", "</s>",
        "<|im_end|>", "<|im_start|>", "<|user1|>", "<|user2|>"]


def load(model_name, token=""):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name, token=token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(token=token or None, attn_implementation="eager")
    if torch.cuda.is_available():
        kw.update(device_map="auto", low_cpu_mem_usage=True)
        for extra in ({"dtype": torch.float16}, {"torch_dtype": torch.float16}, {}):
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **kw, **extra)
                break
            except TypeError:
                continue
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to("cpu")
    model.eval()
    return model, tok


def device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def context(tok, model_name, template, lang, marker, turns, style, gen_len):
    """Drop the oldest pair of turns until it fits. Persona and rules always stay."""
    room = MAX_CTX - gen_len - 16
    work = list(turns)
    while True:
        spec = text.build(model_name, template, lang, marker, work, style)
        if spec["mode"] == "chat":
            msgs = [{"role": "system", "content": spec["system"]},
                    {"role": "user", "content": spec["user"]}]
            try:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            except Exception:
                ids = tok(spec["system"] + "\n\n" + spec["user"] + "\n", return_tensors="pt").input_ids
        else:
            ids = tok(spec["text"], return_tensors="pt").input_ids
        if ids.size(1) <= room or len(work) <= 2:
            return ids
        work = work[2:]


def mask_for(model_name):
    return P.GPT2_MASK if "gpt2" in model_name.lower() else None


@torch.no_grad()
def write(model, tok, codec, ids, cfg, rng, state, payload, roles):
    ids = ids.to(device(model))
    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :].float().cpu().numpy()

    mask = mask_for(cfg["MODEL"])
    stops = [r + ":" for r in roles] + ["\n" + r + ":" for r in roles]
    limit = min(cfg["GEN_LEN"], MAX_CTX - ids.size(1))
    toks, bits, used, ent = [], payload, 0, 0.0

    for _ in range(max(0, limit)):
        p, idx = P.get(logits, cfg["TEMPERATURE"], cfg["TOP_P"], mask)
        ent += P.entropy(p)
        tid, nb, state = codec.encode_step(p, idx, bits, state, rng)
        toks.append(int(tid))
        bits = bits[nb:]
        used += nb

        piece = tok.decode([tid])
        if tid == tok.eos_token_id or any(s in piece for s in STOP):
            break
        if any(s in tok.decode(toks[-12:]) for s in stops):
            break
        nxt = model(torch.tensor([[tid]], device=device(model)), past_key_values=past, use_cache=True)
        past = nxt.past_key_values
        logits = nxt.logits[0, -1, :].float().cpu().numpy()

    return {"toks": toks, "bits": used, "entropy": ent, "state": state,
            "raw": tok.decode(toks)}


@torch.no_grad()
def read(model, tok, codec, ids, toks, cfg, rng, state):
    ids = ids.to(device(model))
    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :].float().cpu().numpy()
    mask = mask_for(cfg["MODEL"])
    got = ""
    for tid in toks:
        p, idx = P.get(logits, cfg["TEMPERATURE"], cfg["TOP_P"], mask)
        if tid not in idx:
            break                          # desync, partial decode
        b, state = codec.decode_step(p, idx, tid, state, rng)
        got += b
        nxt = model(torch.tensor([[tid]], device=device(model)), past_key_values=past, use_cache=True)
        past = nxt.past_key_values
        logits = nxt.logits[0, -1, :].float().cpu().numpy()
    return got, state


def clean(raw, roles):
    t = raw.strip()
    for s in STOP + ["</", "<|", "[PAD]", "<pad>"] + [r + ":" for r in roles]:
        if s in t:
            t = t.split(s)[0]
    return t.strip()
