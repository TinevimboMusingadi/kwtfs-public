"""Sample the pool to quotas, dedup it, and assign splits.

    python -m kwtfs.data.build            # writes data/kwtfs/items.jsonl + build_report.json

Dedup runs on the whole sampled pool *before* splitting, so no near-duplicate
can sit on both sides of a split. The OOD split is held out by construction:
one entire bev domain plus 6-digit arithmetic, which training never sees.
"""

from __future__ import annotations

import collections
import json
import pathlib
import random

from kwtfs.data import dedup, schema, sources

OUT = sources.ROOT / "data" / "kwtfs"
OOD_DOMAIN = "Game-state decisions"  # whole bev domain never seen in training
QUOTAS = {  # source -> items sampled before dedup/audit (~12.5k total)
    "bev-decision": 4000, "gsm8k": 2000, "aqua_rat": 1000,
    "arc": 800, "openbookqa": 800, "commonsenseqa": 800,
    "proc-arith": 1600, "proc-compare": 1200, "proc-rubric": 480,
}
TEST, VAL = 804, 300  # split equally over the 6 genres (134 / 50 each)
RL_SHARE = 3 / 8  # of what remains after test/val: SFT 5/8, RL 3/8


def _stratum(it: schema.Item) -> tuple:
  d = it.difficulty
  return (it.genre, it.domain, d.get("digits"), d.get("steps"), d.get("gap"),
          d.get("depth"), d.get("qtype"))


def _sample(items, n, seed):
  """Stratified by domain so small domains survive the quota."""
  by = collections.defaultdict(list)
  for it in items:
    by[it.domain].append(it)
  rng = random.Random(repr(seed))
  for v in by.values():
    rng.shuffle(v)
  out, i = [], 0
  while len(out) < n and any(i < len(v) for v in by.values()):
    for v in by.values():
      if i < len(v) and len(out) < n:
        out.append(v[i])
    i += 1
  return out


def build(seed: int = 0) -> tuple[list[schema.Item], dict]:
  loaded = (sources.bev() + sources.gsm8k() + sources.aqua() + sources.arc()
            + sources.openbookqa() + sources.commonsenseqa() + sources.procedural(n_per_cell=80)
            + _ood_arith())
  loaded = [it for it in loaded if not schema.problems(it)]
  by_source = collections.defaultdict(list)
  for it in loaded:
    by_source[it.source].append(it)

  ood = [it for it in by_source["bev-decision"] if it.domain == OOD_DOMAIN][:300]
  ood += by_source.pop("proc-arith-ood")
  by_source["bev-decision"] = [it for it in by_source["bev-decision"] if it.domain != OOD_DOMAIN]

  picked = []
  for src, quota in QUOTAS.items():
    picked += _sample(by_source[src], quota, (seed, src))
  kept, dropped = dedup.dedup(ood + picked)
  ood_ids = {it.id for it in ood}

  rng = random.Random(seed)
  pool = [it for it in kept if it.id not in ood_ids]
  # Test/val take an equal share per genre (so per-genre numbers have usable
  # n), round-robin over difficulty strata inside each genre.
  by_genre = collections.defaultdict(lambda: collections.defaultdict(list))
  for it in pool:
    by_genre[it.genre][_stratum(it)].append(it)
  ordered = []
  for genre in sorted(by_genre):
    queues = list(by_genre[genre].values())
    for q in queues:
      rng.shuffle(q)
    rng.shuffle(queues)
    seq = []
    while any(queues):
      for q in queues:
        if q:
          seq.append(q.pop())
    per_test, per_val = TEST // len(by_genre), VAL // len(by_genre)
    for i, it in enumerate(seq):
      it.split = "test" if i < per_test else "val" if i < per_test + per_val else None
    ordered += seq
  rest = [it for it in ordered if it.split is None]
  rng.shuffle(rest)
  n_rl = int(len(rest) * RL_SHARE)
  for i, it in enumerate(rest):
    it.split = "rl" if i < n_rl else "sft"
  for it in kept:
    if it.id in ood_ids:
      it.split = "ood"

  src_of = {x.id: x.source for x in ood + picked}
  report = {
      "seed": seed,
      "loaded": dict(collections.Counter(it.source for it in loaded)),
      "dedup_dropped": dict(collections.Counter(src_of[d] for d in dropped)),
      "splits": {s: dict(collections.Counter(it.genre for it in kept if it.split == s))
                 for s in ("sft", "rl", "val", "test", "ood")},
      "total": len(kept),
  }
  return kept, report


def _ood_arith() -> list[schema.Item]:
  """6-digit arithmetic: outside the 1-5 digit range used in training."""
  out = []
  for it in sources.procedural(n_per_cell=40):
    if it.source == "proc-arith" and it.difficulty["digits"] == 5:
      out.append(it)
  shifted = []
  for it in out:
    rng = sources._rng("ood", it.id)  # pylint: disable=protected-access
    a = rng.randint(100000, 999999)
    b = rng.randint(1000, 99999)
    c, d = rng.randint(1, 9), rng.randint(1000, 99999)
    steps = it.difficulty["steps"]
    q, gold = {
        1: (f"A crate holds {a} items. {b} more arrive. How many items are there now?", a + b),
        2: (f"A delivery has {a} boxes with {c} parts each. {b} parts are removed. How many parts are left?", a * c - b),
        3: (f"Two depots each receive {a} parcels and each dispatches {b}. Then {d} more arrive at the first depot. How many parcels are there in total?", 2 * (a - b) + d),
        4: (f"A shop has {a} pens. It sells {b // 10} pens a day for {c} days, then receives {d} more. How many pens does it have?", a - (b // 10) * c + d),
    }[steps]
    shown = rng.random() < 0.5
    options, g = sources._with_options(rng, gold, ("ood", it.id), shown)  # pylint: disable=protected-access
    shifted.append(schema.Item(
        id=it.id.replace("proc-arith", "proc-arith-ood"), source="proc-arith-ood",
        licence="generated", genre="arith", domain=f"arith-{steps}step", question=q,
        options=options, gold=g, answer_type="number",
        difficulty={"digits": 6, "steps": steps, "options_shown": shown}))
  return shifted


def main() -> None:
  items, report = build()
  OUT.mkdir(parents=True, exist_ok=True)
  schema.write(OUT / "items.jsonl", items)
  (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()


def drop_long(path=OUT / "items.jsonl", max_tokens: int = 512, tokenizer_json: str | None = None):
  """Remove items whose longest profiling prompt exceeds the sampler's prompt
  budget (found by the prep smoke run: 74 bev items, all > 512 tokens). Done
  in place on the built file so already-profiled ids stay valid."""
  import glob  # pylint: disable=g-import-not-at-top
  from tokenizers import Tokenizer  # pylint: disable=g-import-not-at-top

  from kwtfs.data import profile  # pylint: disable=g-import-not-at-top

  tokenizer_json = tokenizer_json or glob.glob(
      str(sources.ROOT / ".cache" / "huggingface" / "models--Qwen--Qwen3-1.7B" / "snapshots" / "*" / "tokenizer.json"),
      )[0]
  tok = Tokenizer.from_file(tokenizer_json)
  items = schema.read(path)
  keep, dropped = [], []
  for it in items:
    n = max(len(tok.encode(profile.prompt(it, m)).ids) for m in ("cot", "think"))
    (dropped if n > max_tokens else keep).append(it)
  schema.write(path, keep)
  report_path = OUT / "build_report.json"
  report = json.loads(report_path.read_text())
  report["dropped_long_prompt"] = {
      "max_tokens": max_tokens, "ids": [it.id for it in dropped],
      "by_source_split": dict(collections.Counter(f"{it.source}/{it.split}" for it in dropped))}
  report["total"] = len(keep)
  report_path.write_text(json.dumps(report, indent=2))
  return len(dropped)


def apply_audit(decisions_path, items_path=OUT / "items.jsonl", profile_path=None):
  """Apply reviewed gold fixes / drops (docs/data/audit_decisions.json) to
  the built items, and re-grade stored profile answers for fixed items so the
  labels stay consistent with the corrected gold."""
  from kwtfs.data import profile as prof  # pylint: disable=g-import-not-at-top

  dec = json.loads(pathlib.Path(decisions_path).read_text(encoding="utf-8"))
  items = schema.read(items_path)
  keep = []
  for it in items:
    if it.id in dec["drop"]:
      continue
    if it.id in dec["fix_gold"]:
      it.meta["gold_before_audit"] = it.gold
      it.gold = dec["fix_gold"][it.id]["gold"]
    keep.append(it)
  schema.write(items_path, keep)
  if profile_path:
    by_id = {it.id: it for it in keep}
    lines = []
    for line in pathlib.Path(profile_path).read_text(encoding="utf-8").splitlines():
      r = json.loads(line)
      if r["id"] in dec["drop"]:
        continue
      if r["id"] in dec["fix_gold"] and r.get("answers") is not None:
        it = by_id[r["id"]]
        if r["mode"] == "jev":
          gold = "ABCDEF"[[prof._norm(o) for o in it.options].index(prof._norm(it.gold))]  # pylint: disable=protected-access
          r["correct"] = [a == gold for a in r["answers"]]
        elif r["mode"] in ("direct", "cot", "think"):
          r["correct"] = [prof.grade(it, a) for a in r["answers"]]
      lines.append(json.dumps(r, ensure_ascii=False))
    pathlib.Path(profile_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
  report_path = OUT / "build_report.json"
  report = json.loads(report_path.read_text())
  report["audit"] = {"fixed": sorted(dec["fix_gold"]), "dropped": sorted(dec["drop"])}
  report["total"] = len(keep)
  report_path.write_text(json.dumps(report, indent=2))
  return len(items) - len(keep), len(dec["fix_gold"])
