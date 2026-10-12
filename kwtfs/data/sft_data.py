"""Build SFT examples: obedience (forced mode) + routing (auto) halves.

    python -m kwtfs.data.sft_data --profile P --traces T --out data/kwtfs/sft.jsonl

Obedience: an item rendered in each feasible mode under "Mode: <m>", so the
mode tag really switches reasoning on and off (and forced CoT without options
ends in <answer>, which an earlier run broke).
Routing: under "Mode: auto" the target is the measured suggested mode; ties go
to the cheaper mode (jev < direct < cot).

CoT bodies are self-distilled: the shortest sample that is correct AND whose
arithmetic checks out (kwtfs/checks.py). Each example records how many leading
completion characters are the genre+mode tags, so SFT can upweight them.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re

from kwtfs import checks
from kwtfs import format as fmt
from kwtfs.data import labels, profile, schema

EOS = "<|im_end|>"
COST_ORDER = ("jev", "direct", "cot")
_ANSWER_TAIL = re.compile(r"<answer>.*?</answer>\s*$", re.DOTALL)


def gold_letter(it: schema.Item) -> str:
  norm = [profile._norm(o) for o in it.options]  # pylint: disable=protected-access
  return fmt.LETTERS[norm.index(profile._norm(it.gold))]  # pylint: disable=protected-access


def best_cot_body(it: schema.Item, samples: list[dict]) -> str | None:
  """Shortest correct, faithful, properly closed sample, minus its answer tag."""
  good = [s for s in samples if s["correct"] and "<answer>" in s["text"]
          and checks.faithful(s["text"])]
  if not good:
    return None
  text = min(good, key=lambda s: s["tokens"])["text"]
  body = _ANSWER_TAIL.sub("", text.split("<answer>")[0] + "<answer>x</answer>").strip()
  body = re.sub(r"<mode:\w+>|<genre:\w+>", "", body).strip()  # never leak tags
  return body or None


def render(it: schema.Item, mode: str, cot_body: str | None) -> str | None:
  """Completion (without tags) for a mode, or None if it can't be built."""
  if mode == "jev":
    return fmt.JEV_PREFIX + gold_letter(it) if it.options else None
  if mode == "direct":
    text = f"<mode:direct>\n<answer>{it.gold}</answer>"
    return text if len(text) <= 120 else None  # fmt.parse rejects longer direct
  if cot_body is None:
    return None
  end = f"<mode:jev>\n{gold_letter(it)}" if it.options else f"<answer>{it.gold}</answer>"
  return f"<mode:cot>\n{cot_body}\n{end}"


def routing_target(lab: dict, available: set[str]) -> str | None:
  """Cheapest mode within the tie band of the best measured mode. If that mode
  has no usable completion (e.g. no faithful CoT trace), there is no routing
  example: falling back to a worse mode would teach the wrong route."""
  u = lab["u"]
  best = max(u.values())
  for m in COST_ORDER:
    if m in u and u[m] >= best - labels.TIE:
      return m if m in available else None
  return None


def build(items, profiles, traces, obedience_share=0.5, seed=0, lam=labels.LAMBDA):
  rng = random.Random(seed)
  obey, route = [], []
  stats = collections.Counter()
  for it in items:
    modes = profiles.get(it.id, {})
    if not all(m in modes for m in ("direct", "cot")):
      stats["no_profile"] += 1
      continue
    body = best_cot_body(it, traces.get(it.id, []))
    stats["cot_trace" if body else "no_cot_trace"] += 1
    rendered = {m: render(it, m, body) for m in profile.modes_for(it)}
    rendered = {m: r for m, r in rendered.items() if r}
    tag = f"<genre:{it.genre}>"
    question = fmt.render_question(it.question, it.options)
    for m, r in rendered.items():
      obey.append({"id": it.id, "kind": "obey", "control": m, "mode": m,
                   "prompt": fmt.chat_prompt(question, m),
                   "completion": tag + r + EOS, "tag_chars": len(tag) + len(f"<mode:{m}>")})
    lab = labels.label(labels.utilities(modes, lam))
    target = routing_target(lab, set(rendered))
    if target:
      stats[f"route_{target}"] += 1
      route.append({"id": it.id, "kind": "route", "control": "auto", "mode": target,
                    "prompt": fmt.chat_prompt(question, "auto"),
                    "completion": tag + rendered[target] + EOS,
                    "tag_chars": len(tag) + len(f"<mode:{target}>"),
                    "ambiguous": lab["ambiguous"]})
  # Mix: routing examples all kept; obedience subsampled to the requested share.
  n_obey = int(len(route) * obedience_share / (1 - obedience_share))
  rng.shuffle(obey)
  data = route + obey[:n_obey]
  rng.shuffle(data)
  stats["examples"] = len(data)
  stats["obey"] = min(n_obey, len(obey))
  stats["route"] = len(route)
  by_id = {it.id: it for it in items}
  for ex in data:  # every target must parse as valid under its control
    p = fmt.parse_tagged(ex["completion"].removesuffix(EOS), len(by_id[ex["id"]].options),
                     ex["control"])
    assert p.valid and p.trace.first_mode == ex["mode"], ex
  return data, dict(stats)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--items", default="data/kwtfs/items.jsonl")
  ap.add_argument("--profile", required=True)
  ap.add_argument("--traces", required=True)
  ap.add_argument("--out", default="data/kwtfs/sft.jsonl")
  ap.add_argument("--split", default="sft")
  args = ap.parse_args()
  items = [it for it in schema.read(args.items) if it.split == args.split]
  profiles = labels.load_profiles(args.profile)
  traces = {}
  with open(args.traces, encoding="utf-8") as handle:
    for line in handle:
      r = json.loads(line)
      traces[r["id"]] = r["samples"]
  data, stats = build(items, profiles, traces)
  with open(args.out, "w", encoding="utf-8") as handle:
    for ex in data:
      handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
  print(json.dumps(stats, indent=2))


if __name__ == "__main__":
  main()
