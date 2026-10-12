"""One row format for every source, so audit, profiling and training share it.

An item is a single-turn question with an exact gold answer. When `options`
is non-empty, `gold` is one of them (exactly once) and JEV is available.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

GENRES = ("decision", "compare", "rubric", "arith", "wordproblem", "knowledge")
ANSWER_TYPES = ("letter", "number", "text")
SPLITS = ("pool", "sft", "rl", "val", "test", "ood")
MAX_OPTIONS = 6  # option letters A-F are single Qwen tokens (kwtfs.format.LETTERS)


@dataclasses.dataclass
class Item:
  id: str
  source: str  # e.g. "gsm8k", "bev-decision", "proc-arith"
  licence: str
  genre: str
  domain: str  # finer label inside a genre (bev domain, ARC-Easy/Challenge, ...)
  question: str
  options: list[str]
  gold: str
  answer_type: str
  difficulty: dict = dataclasses.field(default_factory=dict)  # source features
  split: str = "pool"
  meta: dict = dataclasses.field(default_factory=dict)  # upstream ids, notes

  @property
  def provenance_hash(self) -> str:
    key = json.dumps([self.source, self.question, self.options, self.gold])
    return hashlib.sha256(key.encode()).hexdigest()[:16]

  def to_json(self) -> str:
    return json.dumps({**dataclasses.asdict(self), "hash": self.provenance_hash},
                      ensure_ascii=False)

  @classmethod
  def from_json(cls, line: str) -> "Item":
    d = json.loads(line)
    d.pop("hash", None)
    return cls(**d)


def problems(item: Item) -> list[str]:
  """Structural defects. An empty list means the item is usable."""
  out = []
  if item.genre not in GENRES:
    out.append(f"unknown genre {item.genre!r}")
  if item.answer_type not in ANSWER_TYPES:
    out.append(f"unknown answer_type {item.answer_type!r}")
  if item.split not in SPLITS:
    out.append(f"unknown split {item.split!r}")
  if not item.question.strip():
    out.append("empty question")
  if not str(item.gold).strip():
    out.append("empty gold")
  if item.options:
    if len(item.options) > MAX_OPTIONS:
      out.append(f"{len(item.options)} options > {MAX_OPTIONS}")
    norm = [o.strip().lower() for o in item.options]
    if any(not o for o in norm):
      out.append("empty option")
    if len(set(norm)) != len(norm):
      out.append("duplicate options")
    hits = norm.count(str(item.gold).strip().lower())
    if hits != 1:
      out.append(f"gold appears {hits} times in options")
  if item.answer_type == "number":
    try:
      float(str(item.gold).replace(",", ""))
    except ValueError:
      out.append(f"non-numeric gold {item.gold!r}")
  return out


def write(path, items) -> None:
  with open(path, "w", encoding="utf-8") as handle:
    for item in items:
      handle.write(item.to_json() + "\n")


def read(path) -> list[Item]:
  with open(path, encoding="utf-8") as handle:
    return [Item.from_json(line) for line in handle if line.strip()]
