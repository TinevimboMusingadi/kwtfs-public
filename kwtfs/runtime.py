"""Shared inference plumbing for native mode execution (Tunix sampler + JEV letters).

The mode token the model emits changes what runs next:

  <mode:direct>  short generation (answer cap), no reasoning budget
  <mode:cot>     longer generation; a later <mode:jev> hands off to a decision
  <mode:jev>     no generation at all: one forward step, logits restricted to
                 the option letters, argmax + calibrated probabilities

The batched, control-aware executor that does this lives in
`kwtfs.eval.evaluate.Runner`; capability profiling (`kwtfs.data.profile`) uses
the same sampler and letter ids. Needs the `[train]` extra (JAX + Tunix).
"""

from __future__ import annotations

from kwtfs import format as fmt

COT_STEPS = 512
MAX_PROMPT = 1024


class SwitchingRuntime:
  """Holds a greedy Tunix sampler and the single-token ids of the option letters."""

  def __init__(self, model, tokenizer, cache_size: int = MAX_PROMPT + COT_STEPS + 64):
    from tunix.common import configs  # pylint: disable=g-import-not-at-top
    from tunix.generate import sampler as sampler_lib  # pylint: disable=g-import-not-at-top

    self.model = model
    cfg = model.config
    self.sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=configs.CacheConfig(
            cache_size=cache_size,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )
    self.tok = self.sampler.tokenizer
    self.letter_ids = letter_ids(self.tok)


def letter_ids(tok) -> list[int]:
  """Token id of each option letter; JEV needs every letter to be one token."""
  ids = []
  for letter in fmt.LETTERS:
    enc = tok.encode(letter)
    if len(enc) != 1:
      raise ValueError(f"option letter {letter!r} is not a single token: {enc}")
    ids.append(enc[0])
  return ids
