"""Evaluation: every control (auto / forced jev, direct, cot), native execution.

    python -m kwtfs.eval.evaluate --lora in/lora.npz --items in/labelled.jsonl \
        --out out --splits val test [--limit N]
    python -m kwtfs.eval.evaluate --loras sft=in/sft.npz e3b=in/e3b.npz ...   # one base, many adapters
    python -m kwtfs.eval.evaluate --lora in/lora.npz --latency 160 ...      # batch-1 wall clock

Without --lora/--loras the base model is evaluated.

The model writes its genre + mode tags under the given control (so obedience is
measured, not imposed). The runtime then executes the emitted mode natively:
  jev     one forward step (last-position logits only), softmax over letters
  direct  48-token generation
  cot     256-token generation; a <mode:jev> hand-off becomes a JEV step
Caps match profiling and RL, so utilities are comparable with the labels.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import time

import numpy as np

from kwtfs import checks
from kwtfs import format as fmt
from kwtfs.data import labels

HEAD_STEPS, DIRECT_STEPS, COT_STEPS = 16, 48, 256
CACHE = 768 + COT_STEPS + 64
_HEAD = re.compile(r"\s*(?:<genre:(\w+)>)?\s*<mode:(jev|direct|cot)>")


def _log(msg: str) -> None:
  print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Runner:

  def __init__(self, model, tokenizer, batch: int = 16):
    from kwtfs import runtime  # pylint: disable=g-import-not-at-top

    self.rt = runtime.SwitchingRuntime(model, tokenizer, cache_size=CACHE)
    self.model, self.tok, self.batch = model, self.rt.tok, batch

  def generate(self, prompts: list[str], steps: int) -> tuple[list[str], list[int]]:
    texts, counts = [], []
    for s in range(0, len(prompts), self.batch):
      chunk = prompts[s:s + self.batch]
      real = len(chunk)
      chunk = chunk + [chunk[0]] * (self.batch - real)
      out = self.rt.sampler(input_strings=chunk, max_generation_steps=steps,
                            max_prompt_length=768, temperature=0.0,
                            eos_tokens=[self.tok.eos_id()])
      texts += list(out.text)[:real]
      counts += [len(t) for t in out.tokens][:real]
    return texts, counts

  def jev(self, prefixes: list[str], num_options: list[int]) -> list[dict[str, float]]:
    """Next-token distribution over the option letters, last position only."""
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    from tunix.generate import utils as gen_utils  # pylint: disable=g-import-not-at-top

    out = []
    pad = self.tok.pad_id()
    for s in range(0, len(prefixes), self.batch):
      chunk = prefixes[s:s + self.batch]
      real = len(chunk)
      ids = [self.tok.encode(p) for p in chunk]
      ids += [ids[0]] * (self.batch - real)
      length = -(-max(len(i) for i in ids) // 256) * 256
      tokens = jnp.asarray([[pad] * (length - len(i)) + i for i in ids], jnp.int32)
      mask = jnp.asarray([[0] * (length - len(i)) + [1] * len(i) for i in ids], jnp.bool_)
      hidden, _ = self.model(tokens, gen_utils.build_positions_from_mask(mask), None,
                             gen_utils.make_causal_attn_mask(mask, length), skip_lm_head=True)
      logits = self.model.compute_final_logits(hidden[:, -1:, :])[:, 0, :]
      logits = np.asarray(jax.device_get(logits), np.float32)
      for row, n in zip(logits[:real], num_options[s:s + real]):
        scores = row[self.rt.letter_ids[:n]]
        p = np.exp(scores - scores.max())
        p /= p.sum()
        out.append({fmt.LETTERS[i]: float(v) for i, v in enumerate(p)})
    return out

  def run(self, rows: list[dict], control: str) -> list[dict]:
    prompts = [fmt.chat_prompt(fmt.render_question(r["question"], r["options"]), control)
               for r in rows]
    heads, _ = self.generate(prompts, HEAD_STEPS)
    res: list[dict] = [{} for _ in rows]
    todo = collections.defaultdict(list)  # mode -> [(i, prefix, tag_text)]
    for i, (r, head) in enumerate(zip(rows, heads)):
      m = _HEAD.match(head)
      if not m:
        res[i] = {"completion": head, "tokens": HEAD_STEPS}
        continue
      tag = head[:m.end()].lstrip()
      todo[m.group(2)].append((i, prompts[i] + tag + "\n", tag))
    handoff = []
    for mode, steps in (("direct", DIRECT_STEPS), ("cot", COT_STEPS)):
      items = todo.get(mode, [])
      texts, counts = self.generate([p for _, p, _ in items], steps)
      for (i, prefix, tag), text, n in zip(items, texts, counts):
        tag_tokens = len(self.tok.encode(tag))
        if mode == "cot" and "<mode:jev>" in text and rows[i]["options"]:
          body = text[:text.index("<mode:jev>")]
          handoff.append((i, prefix + body + fmt.JEV_PREFIX,
                          tag + "\n" + body + fmt.JEV_PREFIX, tag_tokens + n))
          continue
        res[i] = {"completion": tag + "\n" + text, "tokens": tag_tokens + n}
    jev_items = [(i, p, tag + "\n", len(self.tok.encode(tag)))
                 for i, p, tag in todo.get("jev", []) if rows[i]["options"]]
    for i, _, tag in todo.get("jev", []):
      if not rows[i]["options"]:
        res[i] = {"completion": tag + "\nA", "tokens": len(self.tok.encode(tag)) + 1}
    batch = jev_items + [(i, p, pre, n) for i, p, pre, n in handoff]
    probs = self.jev([p for _, p, _, _ in batch], [len(rows[i]["options"]) for i, *_ in batch])
    for (i, _, pre, n), pr in zip(batch, probs):
      letter = max(pr, key=pr.get)
      # pre is "<genre:g><mode:jev>\n" or "...<mode:cot>\n<body><mode:jev>\n"
      res[i] = {"completion": pre + letter, "tokens": n + 1, "jev_probs": pr}
    return res


def score(rows: list[dict], results: list[dict], control: str, lam: float = labels.LAMBDA):
  per = collections.defaultdict(lambda: collections.defaultdict(list))
  examples = []
  for r, res in zip(rows, results):
    p = fmt.parse_tagged(res["completion"], len(r["options"]), control)
    ok = p.valid and fmt.is_correct(p.trace, str(r["gold"]), r["options"])
    mode = p.trace.first_mode
    lab = r.get("label") or {}
    for key in ("all", r["genre"]):
      d = per[key]
      d["correct"].append(ok)
      d["tokens"].append(res["tokens"])
      d["valid"].append(p.trace.valid)
      d["mode"].append(mode or "none")
      d["genre_ok"].append(p.genre == r["genre"])
      if control != "auto":
        d["obeyed"].append(p.obeyed)
      if "cot" in (p.trace.modes or ()):
        d["faithful"].append(checks.faithful(res["completion"]))
      if control == "auto" and lab and mode in lab.get("u", {}):
        d["regret"].append(max(lab["u"].values()) - lab["u"][mode])
        if not lab["ambiguous"]:
          d["route_ok"].append(mode == lab["suggested"])
    if len(examples) < 8:
      examples.append({"genre": r["genre"], "question": r["question"][:300],
                       "gold": r["gold"], "completion": res["completion"][:600],
                       "correct": ok, "tokens": res["tokens"]})
  out = {}
  for key, d in per.items():
    acc, tok = statistics.mean(d["correct"]), statistics.mean(d["tokens"])
    share = collections.Counter(d["mode"])
    out[key] = {
        "n": len(d["correct"]), "accuracy": round(acc, 4), "mean_tokens": round(tok, 1),
        "utility": round(acc - lam * tok, 4), "valid": round(statistics.mean(d["valid"]), 4),
        "first_mode": {m: round(v / len(d["mode"]), 3) for m, v in share.items()},
        "genre_tag_acc": round(statistics.mean(d["genre_ok"]), 4),
        **({"obedience": round(statistics.mean(d["obeyed"]), 4)} if d["obeyed"] else {}),
        **({"faithful": round(statistics.mean(d["faithful"]), 4)} if d["faithful"] else {}),
        **({"regret": round(statistics.mean(d["regret"]), 4)} if d["regret"] else {}),
        **({"routing_acc": round(statistics.mean(d["route_ok"]), 4),
            "routing_n": len(d["route_ok"])} if d["route_ok"] else {}),
    }
  out["examples"] = examples
  return out


def latency(model, tok, rows: list[dict], out: pathlib.Path) -> dict:
  """Wall-clock per question at batch 1. A warm-up pass compiles every shape
  (prompt bucket, mode, JEV length bucket) so the timed pass measures serving."""
  runner = Runner(model, tok, batch=1)
  out.mkdir(parents=True, exist_ok=True)
  report = {}
  for control in ("auto", "cot"):
    for r in rows:  # warm-up, untimed
      runner.run([r], control)
    times, tokens, modes = [], [], []
    for r in rows:
      t0 = time.perf_counter()
      res = runner.run([r], control)[0]
      times.append(time.perf_counter() - t0)
      tokens.append(res["tokens"])
      m = _HEAD.match(res["completion"])
      modes.append(m.group(2) if m else "none")
    report[control] = {
        "n": len(rows), "mean_s": round(statistics.mean(times), 3),
        "median_s": round(statistics.median(times), 3),
        "p90_s": round(sorted(times)[int(0.9 * len(times))], 3),
        "mean_tokens": round(statistics.mean(tokens), 1),
        "per_mode_mean_s": {m: round(statistics.mean([t for t, mm in zip(times, modes) if mm == m]), 3)
                            for m in set(modes)},
    }
    _log(f"latency {control}: {report[control]}")
  (out / "latency.json").write_text(json.dumps(report, indent=2))
  return report


def table(report: dict) -> str:
  lines = ["| control | acc | tokens | utility | obey | routing acc | regret | modes j/d/c |",
           "|---|---|---|---|---|---|---|---|"]
  for control, rep in report.items():
    a = rep["all"]
    fm = a["first_mode"]
    lines.append(
        f"| {control} | {a['accuracy']:.1%} | {a['mean_tokens']:.1f} | {a['utility']:.3f} | "
        f"{a.get('obedience', float('nan')):.1%} | {a.get('routing_acc', float('nan')):.1%} | "
        f"{a.get('regret', float('nan')):.3f} | "
        f"{fm.get('jev', 0):.0%}/{fm.get('direct', 0):.0%}/{fm.get('cot', 0):.0%} |")
  return "\n".join(lines)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--lora")
  ap.add_argument("--loras", nargs="*", default=[],
                  help="name=path ...: evaluate several adapters on one loaded base "
                  "(outputs go to <out>/<name>/)")
  ap.add_argument("--items", default="in/labelled.jsonl")
  ap.add_argument("--out", required=True)
  ap.add_argument("--splits", nargs="+", default=["val"])
  ap.add_argument("--controls", nargs="+", default=list(fmt.CONTROLS))
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--batch", type=int, default=16)
  ap.add_argument("--model", default="qwen3-4b=Qwen/Qwen3-4B")
  ap.add_argument("--latency", type=int, default=0,
                  help="instead of scoring: time N test items one at a time (batch 1) "
                  "under auto and forced cot, after an untimed warm-up pass")
  ap.add_argument("overrides", nargs="*")
  args = ap.parse_args()

  from kwtfs.train import grpo  # pylint: disable=g-import-not-at-top

  name, hf_id = args.model.split("=")
  model, tok = grpo.load_model(None, grpo.model_overrides(name, hf_id)
                               + ["model_config.dtype=bfloat16"] + args.overrides)
  runs = [kv.split("=", 1) for kv in args.loras] or [("", args.lora)]
  with open(args.items, encoding="utf-8") as handle:
    all_rows = [json.loads(line) for line in handle]
  for name, lora in runs:
    if lora:
      _log(f"{name}: loaded {grpo.load_lora(model, lora)} LoRA tensors from {lora}")
    if args.latency:
      test = [r for r in all_rows if r["split"] == "test"]  # stored grouped by source:
      step = max(1, len(test) // args.latency)              # take every k-th row
      latency(model, tok, test[::step][:args.latency], pathlib.Path(args.out) / name)
      continue
    runner = Runner(model, tok, args.batch)
    out = pathlib.Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
      rows = [r for r in all_rows if r["split"] == split]
      if args.limit:
        rows = rows[:args.limit]
      report = {}
      for control in args.controls:
        sub = [r for r in rows if control not in ("jev",) or r["options"]]
        t0 = time.time()
        results = runner.run(sub, control)
        report[control] = score(sub, results, control)
        with open(out / f"completions_{split}_{control}.jsonl", "w", encoding="utf-8") as handle:
          for r, res in zip(sub, results):  # everything, so scoring can be redone offline
            handle.write(json.dumps({"id": r["id"], **res}, ensure_ascii=False) + "\n")
        _log(f"{name} {split}/{control}: {len(sub)} rows in {time.time() - t0:.0f}s "
             f"acc={report[control]['all']['accuracy']:.3f}")
        with open(out / f"eval_{split}.json", "w", encoding="utf-8") as handle:
          json.dump(report, handle, indent=2)
      _log(f"{name} {split}\n{table(report)}")
  _log("done")


if __name__ == "__main__":
  main()
