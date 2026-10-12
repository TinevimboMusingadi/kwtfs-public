"""Capability profiling (Phase 1.4) and the model-based gold audit (Phase 1.2).

Base Qwen3-4B answers every item in each mode it can use. The per-mode accuracy
and token cost become the "suggested mode" labels (kwtfs/data/labels.py).

  jev     no generation: one forward step, softmax over the shown option letters
  direct  no-think, "final answer only", 32-token cap, K samples at T=0.7
  cot     no-think, "a few short lines, then <answer>", 256-token cap, K samples
  think   (audit only) native Qwen3 thinking, 768-token cap, then a forced
          answer: a JEV read-out for option items, a short <answer> otherwise.
          A confident disagreement with gold flags the item for review.

    python -m kwtfs.data.profile --items in/items.jsonl --out out [--k 4] [--limit N]

Results are appended per batch to <state>/profile.jsonl, where <state> is the
Tunix checkpoint dir that scripts/vm_startup.sh mirrors to GCS, so a preempted
spot VM resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import time

from kwtfs import format as fmt
from kwtfs.data import schema

DIRECT_STEPS, COT_STEPS, THINK_STEPS = 32, 256, 768
PROMPT_BUCKETS = (256, 512)
MAX_PROMPT = PROMPT_BUCKETS[-1]
SYSTEM = "You are a careful assistant."
VOCAB = 151936
JEV_BATCH = 16  # last-position read-out: memory is the forward pass, not the vocab
JEV_LOGIT_BYTES = 1e9  # nominal [B, L, VOCAB] f32 budget; real peak is ~2.75x (3e9 asked for 8.26 GB)
ORDER = ("val", "test", "ood", "audit", "rl", "sft")  # most useful first if preempted

_LETTER = re.compile(r"^\s*\(?([A-F])\)?(?=[\s.):,]|$)")
_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_ANSWER_IS = re.compile(r"answer is[:\s]*\**\(?([^\n]+?)\)?\**\s*\.?\s*$", re.I | re.M)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# -- prompts -------------------------------------------------------------------

def _chat(user: str, think: bool) -> str:
  head = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
          f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
  return head if think else head + "<think>\n\n</think>\n\n"


def prompt(item: schema.Item, mode: str) -> str:
  q = fmt.render_question(item.question, item.options)
  # A concrete example, not a placeholder: with "<answer>FINAL</answer>" the
  # model copied the word FINAL (CoT scored ~0% in the first TPU smoke run).
  if item.options:
    # The example letter rotates per item, so any anchoring on it averages out
    # over positions (gold positions are shuffled independently).
    letter = "ABCDEF"[int(hashlib.sha256(item.id.encode()).hexdigest(), 16) % len(item.options)]
    final, example = "the letter of your chosen option", f"<answer>{letter}</answer>"
  else:
    final, example = "only the final value", "<answer>42</answer>"
  if mode == "jev":
    return _chat(q + "\n\nReply with the letter of the correct option.", False) + "Answer:\n"
  if mode == "direct":
    return _chat(q + f"\n\nGive only the final answer ({final}), with no explanation.", False)
  if mode == "cot":
    return _chat(q + "\n\nThink step by step in a few short lines, then end with "
                 f"{final} inside answer tags, for example {example}.", False)
  if mode == "think":
    return _chat(q + f"\n\nEnd with {final} inside answer tags, for example {example}.", True)
  raise ValueError(mode)


def modes_for(item: schema.Item) -> tuple[str, ...]:
  return ("jev", "direct", "cot") if item.options else ("direct", "cot")


# -- grading -------------------------------------------------------------------

def _norm(text: str) -> str:
  return re.sub(r"\s+", " ", text.strip().strip(".").strip().lower())


def extract(text: str, mode: str) -> str | None:
  if mode == "direct":
    return text.strip().split("\n")[0] if text.strip() else None
  found = _ANSWER.findall(text)
  if found:
    return found[-1]
  m = _ANSWER_IS.findall(text)
  return m[-1] if m else None


def grade(item: schema.Item, answer: str | None) -> bool:
  """Exact, no substring credit: a letter, an option's text, or the number."""
  if answer is None:
    return False
  if item.options:
    m = _LETTER.match(answer)
    if m:
      idx = "ABCDEF".index(m.group(1))
      return idx < len(item.options) and _norm(item.options[idx]) == _norm(item.gold)
    return _norm(answer) == _norm(item.gold)
  if item.answer_type == "number":
    nums = _NUMBER.findall(answer)
    if not nums:
      return False
    try:
      return abs(float(nums[-1].replace(",", "")) - float(item.gold.replace(",", ""))) < 1e-6
    except ValueError:
      return False
  return _norm(answer) == _norm(item.gold)


def audit_set(items: list[schema.Item], per_source: int = 100) -> list[schema.Item]:
  """Every val/test/ood item (their labels matter most) + a sample of the noisy
  human-labelled sources from the training pools."""
  chosen = [it for it in items if it.split in ("val", "test", "ood")]
  noisy = ("aqua_rat", "bev-decision", "arc", "openbookqa", "commonsenseqa", "gsm8k")
  for src in noisy:
    pool = [it for it in items if it.source == src and it.split in ("sft", "rl")]
    chosen += pool[:per_source]
  return chosen


# -- runner --------------------------------------------------------------------

class Profiler:

  def __init__(self, model, tokenizer, batch: int, think_batch: int):
    from tunix.common import configs  # pylint: disable=g-import-not-at-top
    from tunix.generate import sampler as sampler_lib  # pylint: disable=g-import-not-at-top

    from kwtfs import runtime  # pylint: disable=g-import-not-at-top

    self.rt = runtime.SwitchingRuntime(model, tokenizer, cache_size=MAX_PROMPT + COT_STEPS + 64)
    self.tok = self.rt.tok
    cfg = model.config
    self.think_sampler = sampler_lib.Sampler(
        transformer=model, tokenizer=tokenizer,
        cache_config=configs.CacheConfig(
            cache_size=MAX_PROMPT + THINK_STEPS + 64, num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim))
    self.batch, self.think_batch = batch, think_batch
    self.calls = 0

  def jev_probs(self, prefixes: list[str], num_options: list[int]) -> list[dict]:
    """Option-letter distribution from the LAST position only: hidden states
    (skip_lm_head) then the LM head on one position. runtime._jev builds f32
    logits for every position ([B, L, 152k]), which OOMed and fragmented HBM
    (9.3 GB asked in smoke run 2; decoder starved in the audit)."""
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    import numpy as np  # pylint: disable=g-import-not-at-top
    from tunix.generate import utils as gen_utils  # pylint: disable=g-import-not-at-top

    out, size, pad = [], JEV_BATCH, self.tok.pad_id()
    for s in range(0, len(prefixes), size):
      chunk = [self.tok.encode(p) for p in prefixes[s:s + size]]
      real = len(chunk)
      chunk += [chunk[0]] * (size - real)
      length = -(-max(len(i) for i in chunk) // 256) * 256
      tokens = jnp.asarray([[pad] * (length - len(i)) + i for i in chunk], jnp.int32)
      mask = jnp.asarray([[0] * (length - len(i)) + [1] * len(i) for i in chunk], jnp.bool_)
      model = self.rt.model
      hidden, _ = model(tokens, gen_utils.build_positions_from_mask(mask), None,
                        gen_utils.make_causal_attn_mask(mask, length), skip_lm_head=True)
      logits = np.asarray(jax.device_get(
          model.compute_final_logits(hidden[:, -1:, :])[:, 0, :]), np.float32)
      for row, n in zip(logits[:real], num_options[s:s + real]):
        scores = row[self.rt.letter_ids[:n]]
        prob = np.exp(scores - scores.max())
        prob /= prob.sum()
        out.append({fmt.LETTERS[i]: float(v) for i, v in enumerate(prob)})
    return out

  def _bucket(self, prompts: list[str]) -> int:
    n = max(len(self.tok.encode(p)) for p in prompts)
    for b in PROMPT_BUCKETS:
      if n <= b:
        return b
    return PROMPT_BUCKETS[-1]

  def generate(self, sampler, prompts, steps, temperature, size):
    """Fixed batch size (pads with copies) so each shape compiles once."""
    real = len(prompts)
    prompts = prompts + [prompts[0]] * (size - real)
    self.calls += 1
    out = sampler(input_strings=prompts, max_generation_steps=steps,
                  max_prompt_length=self._bucket(prompts), temperature=temperature,
                  top_p=1.0 if temperature else None, top_k=50 if temperature else None,
                  seed=self.calls, eos_tokens=[self.tok.eos_id()])
    return list(out.text)[:real], [len(t) for t in out.tokens][:real]

  def sampled(self, items, mode, k):
    steps = DIRECT_STEPS if mode == "direct" else COT_STEPS
    prompts = [prompt(it, mode) for it in items for _ in range(k)]
    recs = []
    for s in range(0, len(prompts), self.batch):
      texts, counts = self.generate(self.rt.sampler, prompts[s:s + self.batch], steps, 0.7,
                                    self.batch)
      recs += list(zip(texts, counts))
    out = []
    for i, it in enumerate(items):
      chunk = recs[i * k:(i + 1) * k]
      answers = [extract(t, mode) for t, _ in chunk]
      out.append({"id": it.id, "mode": mode, "k": k,
                  "correct": [grade(it, a) for a in answers],
                  "tokens": [c for _, c in chunk], "answers": answers,
                  "sample": chunk[0][0][:600]})
    return out

  def jev(self, items):
    order = sorted(range(len(items)), key=lambda i: -len(items[i].question))
    items = [items[i] for i in order]  # longest first: chunk size set by the longest
    probs = self.jev_probs([prompt(it, "jev") for it in items], [len(it.options) for it in items])
    out = []
    for it, pr in zip(items, probs):
      gold = "ABCDEF"[[_norm(o) for o in it.options].index(_norm(it.gold))]
      best = max(pr, key=pr.get)
      out.append({"id": it.id, "mode": "jev", "probs": pr, "p_gold": pr[gold],
                  "correct": [best == gold], "tokens": [1], "answers": [best]})
    return out

  def think_gen(self, items):
    """Native thinking (greedy), stored raw. Answers are read out in a separate
    pass so the thinking decoder and the JEV forward never share the chip."""
    heads = [prompt(it, "think") for it in items]
    out = []
    for s in range(0, len(heads), self.think_batch):
      t, c = self.generate(self.think_sampler, heads[s:s + self.think_batch], THINK_STEPS,
                           0.0, self.think_batch)
      for it, text, n in zip(items[s:s + self.think_batch], t, c):
        out.append({"id": it.id, "mode": "think_gen", "text": text, "tokens": [n],
                    "correct": [], "finished": "</think>" in text})
    return out

  def think(self, items, gen: dict[str, dict]):
    """Forced answer after the stored thinking (budget-forced if it ran long)."""
    out, jev_items, jev_prefix, open_items, open_prefix = [], [], [], [], []
    for it in items:
      g = gen[it.id]
      text, finished = g["text"], g["finished"]
      head = prompt(it, "think")
      thought = text.split("</think>")[0]
      rec = {"id": it.id, "mode": "think", "tokens": g["tokens"], "finished": finished,
             "sample": text[-600:]}
      out.append(rec)
      if it.options:
        jev_items.append(rec)
        jev_prefix.append((it, head + thought + "\n</think>\n\nAnswer:\n"))
      else:
        open_items.append(rec)
        open_prefix.append((it, head + thought + "\n</think>\n\n<answer>"))
    if jev_prefix:
      probs = self.jev_probs([p for _, p in jev_prefix],
                             [len(it.options) for it, _ in jev_prefix])
      for rec, (it, _), pr in zip(jev_items, jev_prefix, probs):
        best = max(pr, key=pr.get)
        rec.update(probs=pr, answers=[best], correct=[grade(it, best)],
                   confidence=pr[best])
    # Prompt + thinking can pass 1024 tokens: use the thinking sampler's cache.
    for s in range(0, len(open_prefix), self.think_batch):
      chunk = open_prefix[s:s + self.think_batch]
      t, _ = self.generate(self.think_sampler, [p for _, p in chunk], 24, 0.0, self.think_batch)
      for rec, (it, _), ans in zip(open_items[s:s + self.think_batch], chunk, t):
        ans = ans.split("</answer>")[0].strip()
        rec.update(answers=[ans], correct=[grade(it, ans)], confidence=None)
    return out


def _state_dir(overrides: list[str], out: pathlib.Path) -> pathlib.Path:
  for o in overrides:
    if o.startswith("rl_training_config.checkpoint_root_directory="):
      return pathlib.Path(o.split("=", 1)[1])
  return out


def _log(msg: str) -> None:
  print(f"[profile {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--items", default="in/items.jsonl")
  ap.add_argument("--out", required=True)
  ap.add_argument("--k", type=int, default=4)
  ap.add_argument("--batch", type=int, default=32)
  ap.add_argument("--think-batch", type=int, default=16)
  ap.add_argument("--chunk", type=int, default=64, help="items per saved chunk")
  ap.add_argument("--limit", type=int, default=0, help="items per split (smoke)")
  ap.add_argument("--skip-audit", action="store_true")
  ap.add_argument("--random-weights", action="store_true", help="local plumbing test")
  ap.add_argument("--model", default="qwen3-4b=Qwen/Qwen3-4B")
  ap.add_argument("overrides", nargs="*")
  args = ap.parse_args()

  from kwtfs.train import grpo  # pylint: disable=g-import-not-at-top

  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  state = _state_dir(args.overrides, out)
  state.mkdir(parents=True, exist_ok=True)
  log_path = state / "profile.jsonl"
  done = set()
  if log_path.exists():
    for line in log_path.open(encoding="utf-8"):
      r = json.loads(line)
      done.add((r["id"], r["mode"]))
  _log(f"resuming with {len(done)} (item, mode) results already saved")

  items = schema.read(args.items)
  by_split = {s: [it for it in items if it.split == s] for s in ORDER if s != "audit"}
  by_split["audit"] = [] if args.skip_audit else audit_set(items)
  if args.limit:
    by_split = {s: v[:args.limit] for s, v in by_split.items()}

  if args.random_weights:
    grpo._random_weights_patch()  # pylint: disable=protected-access
  name, hf_id = args.model.split("=")
  # bf16 compute also makes the sampler's KV cache bf16; Tunix defaults to f32,
  # which OOMs Qwen3-4B on one 16 GB v5e chip (15.94G of 15.75G at batch 32).
  dtype = [] if any(o.startswith("model_config.dtype=") for o in args.overrides) else [
      "model_config.dtype=bfloat16"]
  model, tokenizer = grpo.load_model(
      None, grpo.model_overrides(name, hf_id) + dtype + args.overrides)
  prof = Profiler(model, tokenizer, args.batch, args.think_batch)
  t0, n_done = time.time(), 0

  import gc  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top

  # Mode-major: one kind of program on the chip at a time, compiled programs
  # dropped between modes. Cheap modes first; the long CoT pass is resumable.
  plan = [(m, s) for m in ("jev", "direct", "cot") for s in ORDER if s != "audit"]
  # Read-outs: option items (JEV step) and open items (short decode) in separate
  # passes, with compiled programs cleared in between (mixing them OOMed).
  by_split["audit_opt"] = [it for it in by_split["audit"] if it.options]
  by_split["audit_open"] = [it for it in by_split["audit"] if not it.options]
  plan += [("think_gen", "audit"), ("think", "audit_opt"), ("think", "audit_open")]
  with log_path.open("a", encoding="utf-8") as log:
    last_mode = None
    for mode, split in plan:
      todo = [it for it in by_split[split] if (it.id, mode) not in done
              and (mode.startswith("think") or mode in modes_for(it))]
      if not todo:
        continue
      if (mode, split if mode == "think" else None) != last_mode:
        jax.clear_caches()
        gc.collect()
        last_mode = (mode, split if mode == "think" else None)
      if mode == "think":
        gen = {}
        for line in log_path.open(encoding="utf-8"):
          r = json.loads(line)
          if r["mode"] == "think_gen":
            gen[r["id"]] = r
        todo = [it for it in todo if it.id in gen]
      todo.sort(key=lambda it: len(it.question))  # similar lengths batch together
      _log(f"{split}/{mode}: {len(todo)} items to go")
      for s in range(0, len(todo), args.chunk):
        chunk = todo[s:s + args.chunk]
        if mode == "jev":
          recs = prof.jev(chunk)
        elif mode == "think_gen":
          recs = prof.think_gen(chunk)
        elif mode == "think":
          recs = prof.think(chunk, gen)
        else:
          recs = prof.sampled(chunk, mode, args.k)
        for r in recs:
          r["split"] = split
          log.write(json.dumps(r, ensure_ascii=False) + "\n")
          done.add((r["id"], r["mode"]))
        log.flush()
        n_done += len(chunk)
        acc = sum(sum(r["correct"]) / max(1, len(r["correct"])) for r in recs) / len(recs)
        _log(f"{split}/{mode} {s + len(chunk)}/{len(todo)} acc={acc:.2f} "
             f"rate={n_done / (time.time() - t0):.2f} item-modes/s")

  (out / "profile.jsonl").write_bytes(log_path.read_bytes())
  _log(f"done: {len(done)} results -> {out / 'profile.jsonl'}")


if __name__ == "__main__":
  main()
