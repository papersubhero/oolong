# Testing More Models

In this part we test more models to see whether parameter size or modernity can solve the issues we find in other experiments. 
We provide a general interface to invoke two samplers: **vanilla** (no payload, the control) and
**Discop**.
The vanilla one is to isolate failures on Discop preserving the cover distribution. 

## Layout

```
stego/   codecs. base.py is the interface, probs.py the shared front end
run/     gen.py generates and decodes token by token, main.py drives the chat
cfg/     make.py builds run configs, text.py builds prompts, *.json the assets
eval/    judge.py labels conversations, kappa.py scores agreement
ana/     join.py, stats.py, figs.py
```

## Run

```bash
pip install -r requirements.txt

python cfg/make.py --exp pilot
python run/main.py --cfg cfg/out/pilot__Discop__Qwen-Qwen3.5-9B.csv
```

On a cluster:

```bash
VENV=.venv HF_TOKEN=... bash slurm/all.sh e1
sbatch slurm/judge.sbatch
python ana/stats.py --csv res/all.csv --judge gpt-4o
```

## Notes

Two things bite, and both are quiet.

**Logit scale is arbitrary.** Clamping raw logits to a fixed range flattens the
distribution on any model that sits outside it. GPT-2 lives at [-189, -148], so a
[-50, 50] clamp ties the whole vocabulary and the model samples uniformly. Run
`test/scan.py` on every model before you add it.

**Sender and receiver share one keystream.** They must draw from it identically,
including after the payload runs out, which is most of a 200 token message.
`test/drain.py` is the only test that reaches that path.

Other choices worth knowing: probabilities are float64; attention is eager;
`--channel text` re-tokenises the visible message instead of passing token ids,
which is what a real receiver would have to do; `top_p=1.0` keeps tokens down to
1e-8 mass, below fp16 logit noise, so the nucleus stays tractable.

## Interface

Every codec sees the same thing: `probs` and `indices`, sorted descending, float64,
already through temperature and top_p. It returns a token and how many payload bits
it consumed.

```python
encode_step(probs, indices, bits, state, rng) -> (token_id, n_bits, state)
decode_step(probs, indices, token_id, state, rng) -> (bits, state)
```

`state` is threaded across tokens and across turns. Discop and Meteor ignore it.
SparSamp, Shimmer and iMEC carry a chunk, an interval and a belief respectively.

`rng` is the shared keystream. The runner snapshots it before generating and restores
it before decoding, so both sides replay the same draws. Any asymmetry in how many
draws a codec makes is a desync, and it will not show up in a round trip test that
stops when the payload ends.

## n_bits

Payload bits consumed. Not channel capacity. Discop's tree branches at points where
no payload bit is left; those branch points are capacity, not payload, and counting
them made Discop look ten times wider than the rest.

Capacity is measured separately: `gen.write` accumulates the Shannon entropy of every
step, and `ana/stats.py` reports utilisation as bits over entropy. That number is
comparable across schemes.

## Front end

`probs.get` is the only place logits become probabilities, so sender and receiver
cannot drift.

No clamp. Raw logit scale differs per model and clamping to a fixed window flattens
whatever falls outside it. Subtracting the max before the exponential is the only
stabilisation needed, and it is scale invariant.

`floor` drops tokens under 1e-8. It binds only at `top_p=1.0`, where it takes the
nucleus from ~15k to ~7k tokens with no change to entropy at three decimals.

## Truncation and security

Under `top_p<1` every scheme preserves the *truncated* distribution, not the model's.
A warden holding the model can see the truncation. `top_p=1.0` (e6) is the arm where
security is exact; expect the text to get worse. That trade-off is the point.

## Channel

`--channel ids` hands the receiver the sender's token ids. Idealised, and the usual
assumption. `--channel text` re-tokenises the visible message, which is what a real
receiver gets. The gap between the two is a result, not a bug.

## Context

The seed conversation is history, not part of the system prompt. Truncation drops the
oldest pair of turns and always keeps the persona and the rules. Putting the seed in
the system prompt and then truncating it removes the role, which looks like a model
failure and is not.