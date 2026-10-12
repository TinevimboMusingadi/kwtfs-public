"""Load every data source into `schema.Item` rows.

    python -m kwtfs.data.sources --out data/kwtfs/pool.jsonl

Raw downloads live in data/kwtfs/raw/ (see docs/data/SOURCES.md). Only
train splits of public benchmarks are used: Qwen3 has probably seen their test
splits, and our own sealed test is carved out of this pool later.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re

from kwtfs.data import schema
from kwtfs.data.schema import Item

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "kwtfs" / "raw"
MAX_STATE_CHARS = 1200  # keeps prompts under ~400 tokens


def _rng(*key) -> random.Random:
  return random.Random(hashlib.sha256(repr(key).encode()).hexdigest())


def _shuffle_options(options: list[str], gold: str, key) -> list[str]:
  """Deterministic shuffle, so the gold letter is uniform over positions."""
  opts = list(options)
  _rng("opts", key).shuffle(opts)
  assert gold in opts
  return opts


def _parquet(path):
  import pandas as pd  # pylint: disable=g-import-not-at-top
  return pd.read_parquet(path)


# -- bev-decision: the JEV-native source ------------------------------------

def bev(limit_per_domain: int | None = None) -> list[Item]:
  """One question per state; choice / yes-no / score all become letters."""
  df = _parquet(RAW / "bev-decision" / "data" / "train.parquet")
  items, per_domain = [], collections.Counter()
  for idx, (state, qjson, domain) in enumerate(zip(df.state, df.questions_json, df.domain)):
    if len(state) > MAX_STATE_CHARS:
      continue
    questions = json.loads(qjson)
    name = sorted(questions)[_rng("bevq", idx).randrange(len(questions))]
    q = questions[name]
    if q["type"] == "noul":
      options, gold = ["yes", "no"], "yes" if q["label"] else "no"
    elif q["type"] == "choice":
      options, gold = list(q["criteria"].values()), q["criteria"][q["label"]]
    elif q["type"] == "score":
      options, gold = list(q["criteria"]), q["criteria"][q["label"]]
    else:
      continue
    if len(options) > schema.MAX_OPTIONS:
      continue
    if q["type"] != "score":  # score levels keep their natural order
      options = _shuffle_options(options, gold, ("bev", idx))
    if limit_per_domain and per_domain[domain] >= limit_per_domain:
      continue
    per_domain[domain] += 1
    items.append(Item(
        id=f"bev-{idx}-{name}", source="bev-decision", licence="mixed/unknown",
        genre="decision", domain=domain,
        question=f"{state.strip()}\n\n{q['instructions'].strip()}",
        options=options, gold=gold, answer_type="text",
        difficulty={"qtype": q["type"], "n_options": len(options),
                    "state_chars": len(state)},
        meta={"row": idx, "question_name": name}))
  return items


# -- GSM8K and AQuA-RAT: multi-step word problems ---------------------------

_GSM_FINAL = re.compile(r"####\s*(.+)$")


def gsm8k() -> list[Item]:
  df = _parquet(RAW / "gsm8k" / "main" / "train-00000-of-00001.parquet")
  items = []
  for idx, (q, a) in enumerate(zip(df.question, df.answer)):
    gold = _GSM_FINAL.search(a.strip()).group(1).replace(",", "").strip()
    steps = len([l for l in a.split("\n") if "<<" in l])
    items.append(Item(
        id=f"gsm8k-train-{idx}", source="gsm8k", licence="MIT",
        genre="wordproblem", domain="gsm8k", question=q.strip(), options=[],
        gold=gold, answer_type="number",
        difficulty={"steps": steps, "chars": len(q)},
        meta={"solution": a}))
  return items


def aqua() -> list[Item]:
  path = next((RAW / "aqua_rat" / "raw").glob("train-*.parquet"))
  df = _parquet(path)
  items = []
  for idx, row in enumerate(df.itertuples()):
    opts = [re.sub(r"^[A-E]\)\s*", "", o).strip() for o in row.options]
    letter = str(row.correct).strip()
    if len(opts) != 5 or letter not in "ABCDE":
      continue
    gold = opts["ABCDE".index(letter)]
    items.append(Item(
        id=f"aqua-train-{idx}", source="aqua_rat", licence="Apache-2.0",
        genre="wordproblem", domain="aqua", question=row.question.strip(),
        options=_shuffle_options(opts, gold, ("aqua", idx)) if len(set(opts)) == 5 else opts,
        gold=gold, answer_type="text",
        difficulty={"rationale_lines": row.rationale.count("\n") + 1},
        meta={"rationale": row.rationale}))
  return items


# -- knowledge MCQ: CoT is expected to help little here ---------------------

def _mcq(source, licence, domain, df, qcol, choices_col, key_col, stem=None):
  items = []
  for idx, row in enumerate(df.itertuples()):
    choices = getattr(row, choices_col)
    labels, texts = list(choices["label"]), [str(t) for t in choices["text"]]
    key = str(getattr(row, key_col)).strip()
    if key not in labels:
      continue
    gold = texts[labels.index(key)]
    question = getattr(row, qcol)
    if stem:
      question = stem(row)
    items.append(Item(
        id=f"{source}-{domain}-{getattr(row, 'id', idx)}", source=source, licence=licence,
        genre="knowledge", domain=domain, question=str(question).strip(),
        options=_shuffle_options(texts, gold, (source, idx)) if len(set(texts)) == len(texts) else texts,
        gold=gold, answer_type="text", difficulty={"n_options": len(texts)},
        meta={"upstream_id": str(getattr(row, "id", idx))}))
  return items


def arc() -> list[Item]:
  out = []
  for sub in ("ARC-Easy", "ARC-Challenge"):
    path = next((RAW / "ai2_arc" / sub).glob("train-*.parquet"))
    out += _mcq("arc", "CC-BY-SA-4.0", sub, _parquet(path), "question", "choices", "answerKey")
  return out


def openbookqa() -> list[Item]:
  path = next((RAW / "openbookqa" / "main").glob("train-*.parquet"))
  return _mcq("openbookqa", "unknown", "openbookqa", _parquet(path),
              "question_stem", "choices", "answerKey")


def commonsenseqa() -> list[Item]:
  path = next((RAW / "commonsense_qa" / "data").glob("train-*.parquet"))
  return _mcq("commonsenseqa", "MIT", "commonsenseqa", _parquet(path),
              "question", "choices", "answerKey")


# -- procedural: exact answers with a difficulty knob inside each genre -----

def _num(rng: random.Random, digits: int) -> int:
  return rng.randint(10 ** (digits - 1) if digits > 1 else 0, 10 ** digits - 1)


def _distractors(rng: random.Random, gold: int, k: int) -> list[int]:
  """Near misses (off-by-small, digit slips), the kind a direct guess makes."""
  out: set[int] = set()
  scale = max(1, abs(gold) // 1000)
  while len(out) < k:
    d = rng.choice([1, 2, 3, 9, 10, 11, 100]) * rng.choice([1, -1]) * rng.choice([1, scale])
    if gold + d != gold and gold + d >= 0:
      out.add(gold + d)
  return sorted(out)


def _with_options(rng, gold: int, key, shown: bool, k: int = 3):
  if not shown:
    return [], str(gold)
  opts = [str(v) for v in [gold, *_distractors(rng, gold, k)]]
  return _shuffle_options(opts, str(gold), key), str(gold)


def procedural(n_per_cell: int = 40) -> list[Item]:
  """compare / arith / rubric with difficulty varied inside each genre."""
  items = []
  # compare: digits x gap. "5975 vs 5974" is hard to eyeball; "1807 vs 11" is not.
  for digits in (2, 3, 4, 5, 6):
    for gap in ("tiny", "small", "large"):
      for i in range(n_per_cell):
        rng = _rng("compare", digits, gap, i)
        a = _num(rng, digits)
        d = {"tiny": rng.randint(1, 3), "small": rng.randint(4, 10 ** max(1, digits - 2)),
             "large": None}[gap]
        b = a + rng.choice([-1, 1]) * d if d else _num(rng, rng.randint(1, digits))
        if a == b or b < 0:
          continue
        if (a > b) != (i % 2 == 0):  # balance yes/no gold
          a, b = b, a
        noun = rng.choice(["units", "boxes", "points", "tickets"])
        q = f"Are {a} {noun} more than {b} {noun}?"
        gold = "yes" if a > b else "no"
        options = _shuffle_options(["yes", "no"], gold, ("compare", digits, gap, i))
        items.append(Item(
            id=f"proc-compare-{digits}-{gap}-{i}", source="proc-compare", licence="generated",
            genre="compare", domain="compare", question=q, options=options, gold=gold,
            answer_type="text", difficulty={"digits": digits, "gap": gap}))
  # arith: digits x steps; options shown for half of them.
  templates = {
      1: ("A crate holds {a} items. {b} more arrive. How many items are there now?",
          lambda a, b, c, d: a + b),
      2: ("A delivery has {a} boxes with {b} parts each. {c} parts are removed. How many parts are left?",
          lambda a, b, c, d: a * b - c),
      3: ("Two depots each receive {a} parcels and each dispatches {b}. Then {c} more arrive at the first depot. How many parcels are there in total?",
          lambda a, b, c, d: 2 * (a - b) + c),
      4: ("A shop has {a} pens. It sells {b} pens a day for {c} days, then receives {d} more. How many pens does it have?",
          lambda a, b, c, d: a - b * c + d),
  }
  for digits in (1, 2, 3, 4, 5):
    for steps, (tmpl, fn) in templates.items():
      for i in range(n_per_cell):
        rng = _rng("arith", digits, steps, i)
        a = _num(rng, digits) + (10 if digits == 1 else 0)
        b = max(1, _num(rng, max(1, digits - 1)))
        c = rng.randint(1, 9)
        d = _num(rng, max(1, digits - 1))
        if steps == 4:
          b = max(1, min(b, a // (c + 1)))
        gold = fn(a, b, c, d)
        if gold < 0:
          continue
        shown = rng.random() < 0.5
        options, g = _with_options(rng, gold, ("arith", digits, steps, i), shown)
        items.append(Item(
            id=f"proc-arith-{digits}-{steps}-{i}", source="proc-arith", licence="generated",
            genre="arith", domain=f"arith-{steps}step", question=tmpl.format(a=a, b=b, c=c, d=d),
            options=options, gold=g, answer_type="number",
            difficulty={"digits": digits, "steps": steps, "options_shown": shown}))
  # rubric: apply a stated scoring rule; nesting depth is the knob. The target
  # score is drawn uniformly first, so no level (e.g. the clipped 0 or 5) dominates.
  for depth in (1, 2, 3):
    for i in range(n_per_cell * 2):
      rng = _rng("rubric", depth, i)
      base = rng.randint(100, 9000)
      if depth == 1:
        rule, levels = f"score = 1 if items >= {base} else 0", ["0", "1"]
        score = i % 2
        x = base + (rng.randint(0, 20) if score else -rng.randint(1, 20))
      else:
        k = 3 if depth == 2 else rng.randint(2, 4)
        off = 0 if depth == 2 else 1
        rule = (f"score = min(5, max(0, (items - {base}) // 3))" if depth == 2 else
                f"score = min(5, max(0, ((items - {base}) // {k}) - 1))")
        levels = [str(v) for v in range(6)]
        score = i % 6
        x = base + (score + off) * k + rng.randrange(k)  # lands exactly in that bucket
        if score == 0:
          x = base + rng.randint(-30, off * k + k - 1)
        elif score == 5:
          x += rng.randint(0, 30)
        got = min(5, max(0, (x - base) // k - off))
        assert got == score, (depth, x, base, k, got, score)
      items.append(Item(
          id=f"proc-rubric-{depth}-{i}", source="proc-rubric", licence="generated",
          genre="rubric", domain=f"rubric-depth{depth}",
          question=f"Using {rule}, score {x} items.", options=levels, gold=str(score),
          answer_type="text", difficulty={"depth": depth, "margin": x - base}))
  return items


LOADERS = {
    "bev": lambda: bev(limit_per_domain=300),
    "gsm8k": gsm8k, "aqua": aqua, "arc": arc,
    "openbookqa": openbookqa, "commonsenseqa": commonsenseqa,
    "procedural": procedural,
}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=str(ROOT / "data" / "kwtfs" / "pool.jsonl"))
  ap.add_argument("--only", nargs="*", default=list(LOADERS))
  args = ap.parse_args()
  items, rejected = [], collections.Counter()
  for name in args.only:
    loaded = LOADERS[name]()
    ok = [it for it in loaded if not schema.problems(it)]
    for it in loaded:
      for p in schema.problems(it):
        rejected[(it.source, p.split(" ")[0] if "gold appears" not in p else p)] += 1
    print(f"{name:14s} loaded={len(loaded):6d} usable={len(ok):6d}")
    items += ok
  schema.write(args.out, items)
  counts = collections.Counter((it.genre, it.source) for it in items)
  for key, n in sorted(counts.items()):
    print(f"  {key[0]:12s} {key[1]:16s} {n}")
  if rejected:
    print("rejected:", dict(rejected))
  print(f"wrote {len(items)} items -> {args.out}")


if __name__ == "__main__":
  main()
