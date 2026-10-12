"""Exact and near-duplicate removal with a small numpy MinHash + LSH.

Near duplicates matter twice: they inflate a source, and across splits they
leak test answers into training. Two items are near duplicates when their
word-5-gram Jaccard similarity is >= THRESHOLD *and* their gold answers match;
the same text with a different answer (procedural variants) is kept.
"""

from __future__ import annotations

import collections
import hashlib
import re

import numpy as np

NUM_PERM = 64
BANDS = 16  # 16 bands x 4 rows: ~50% catch rate at Jaccard ~0.5, ~99% at 0.8
THRESHOLD = 0.8
_PRIME = (1 << 61) - 1
_rng = np.random.default_rng(0)
_A = _rng.integers(1, _PRIME, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, _PRIME, NUM_PERM, dtype=np.uint64)


def normalize(text: str) -> str:
  return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def shingles(text: str, n: int = 5) -> set[int]:
  words = normalize(text).split()
  grams = [" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))]
  return {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
          for g in grams}


def signature(sh: set[int]) -> np.ndarray:
  x = np.fromiter(sh, dtype=np.uint64)
  # (a*x + b) mod p, vectorised over permutations; wraparound is fine for hashing
  return ((np.outer(_A, x) + _B[:, None]) % _PRIME).min(axis=1)


def jaccard(a: set[int], b: set[int]) -> float:
  return len(a & b) / max(1, len(a | b))


def dedup(items, key=lambda it: it.question, answer=lambda it: str(it.gold)):
  """Return (kept, dropped) where dropped maps a removed id -> kept id."""
  kept, dropped = [], {}
  exact: dict[tuple[str, str], str] = {}
  buckets: dict[tuple[int, bytes], list[int]] = collections.defaultdict(list)
  kept_sh: list[set[int]] = []
  rows = NUM_PERM // BANDS
  for it in items:
    norm = (normalize(key(it)), answer(it).strip().lower())
    if norm in exact:
      dropped[it.id] = exact[norm]
      continue
    sh = shingles(key(it))
    sig = signature(sh)
    bands = [(b, sig[b * rows:(b + 1) * rows].tobytes()) for b in range(BANDS)]
    match = None
    for band in bands:
      for j in buckets.get(band, ()):
        if answer(kept[j]).strip().lower() == norm[1] and jaccard(sh, kept_sh[j]) >= THRESHOLD:
          match = kept[j].id
          break
      if match:
        break
    if match:
      dropped[it.id] = match
      continue
    exact[norm] = it.id
    idx = len(kept)
    kept.append(it)
    kept_sh.append(sh)
    for band in bands:
      buckets[band].append(idx)
  return kept, dropped
