"""TPU prep job after profiling: probe features + self-distilled CoT traces.

    python -m kwtfs.data.prep --items in/items.jsonl --out out [--k 4]

1. hidden.npz: Qwen3-4B's final hidden state at the end of the auto prompt,
   for every item (features for kwtfs/mode_value.py).
2. traces.jsonl: K brief-CoT samples (full text) for every SFT-split item. The
   SFT builder keeps the shortest one that is correct and whose arithmetic
   checks out (kwtfs/checks.py). The profile only kept a 600-char sample.

Traces are appended to <state>/traces.jsonl (mirrored to GCS), so a preempted
spot VM resumes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from kwtfs.data import profile, schema


def _log(msg: str) -> None:
  print(f"[prep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--items", default="in/items.jsonl")
  ap.add_argument("--out", required=True)
  ap.add_argument("--k", type=int, default=4)
  ap.add_argument("--batch", type=int, default=32)
  ap.add_argument("--chunk", type=int, default=64)
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--splits", nargs="+", default=["sft"])
  ap.add_argument("--model", default="qwen3-4b=Qwen/Qwen3-4B")
  ap.add_argument("overrides", nargs="*")
  args = ap.parse_args()

  import gc  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top

  from kwtfs import mode_value  # pylint: disable=g-import-not-at-top
  from kwtfs.train import grpo  # pylint: disable=g-import-not-at-top

  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  state = profile._state_dir(args.overrides, out)  # pylint: disable=protected-access
  state.mkdir(parents=True, exist_ok=True)
  items = schema.read(args.items)
  hidden_items = items[:args.limit] if args.limit else items  # --limit: smoke runs

  name, hf_id = args.model.split("=")
  model, tok = grpo.load_model(None, grpo.model_overrides(name, hf_id)
                               + ["model_config.dtype=bfloat16"] + args.overrides)

  hidden_path = state / "hidden.npz"
  if not hidden_path.exists():
    _log(f"hidden states for {len(hidden_items)} items")
    H = mode_value.extract(hidden_items, model, tok, batch=args.batch)
    np.savez_compressed(hidden_path, H=H, ids=np.array([it.id for it in hidden_items]))
  (out / "hidden.npz").write_bytes(hidden_path.read_bytes())
  jax.clear_caches()
  gc.collect()

  log_path = state / "traces.jsonl"
  done = set()
  if log_path.exists():
    done = {json.loads(line)["id"] for line in log_path.open(encoding="utf-8")}
  todo = [it for it in items if it.split in args.splits and it.id not in done]
  if args.limit:
    todo = todo[:args.limit]
  todo.sort(key=lambda it: len(it.question))
  _log(f"traces: {len(todo)} items to go ({len(done)} done)")
  prof = profile.Profiler(model, tok, args.batch, think_batch=1)
  t0 = time.time()
  with log_path.open("a", encoding="utf-8") as log:
    for s in range(0, len(todo), args.chunk):
      chunk = todo[s:s + args.chunk]
      prompts = [profile.prompt(it, "cot") for it in chunk for _ in range(args.k)]
      texts, counts = [], []
      for b in range(0, len(prompts), args.batch):
        t, c = prof.generate(prof.rt.sampler, prompts[b:b + args.batch], profile.COT_STEPS,
                             0.7, args.batch)
        texts += t
        counts += c
      for i, it in enumerate(chunk):
        samples = []
        for text, n in zip(texts[i * args.k:(i + 1) * args.k], counts[i * args.k:(i + 1) * args.k]):
          ans = profile.extract(text, "cot")
          samples.append({"text": text, "tokens": n, "answer": ans,
                          "correct": profile.grade(it, ans)})
        log.write(json.dumps({"id": it.id, "samples": samples}, ensure_ascii=False) + "\n")
      log.flush()
      _log(f"traces {s + len(chunk)}/{len(todo)} "
           f"rate={(s + len(chunk)) / (time.time() - t0):.2f} items/s")
  (out / "traces.jsonl").write_bytes(log_path.read_bytes())
  _log("done")


if __name__ == "__main__":
  main()
