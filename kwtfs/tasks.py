"""RL prompts for the Tunix CLI (`data_module: kwtfs.tasks`).

Rows come from a labelled RL-split file (data/kwtfs/rl.jsonl, staged to the VM
as in/rl.jsonl) written by `kwtfs.data.labels --write`, optionally with
mode-value utilities. 30% of prompts force a mode, so obedience cannot decay
while RL optimises auto routing.

Columns consumed by kwtfs/rewards.py: answer, options, control, genre,
utilities.

Environment variables:
  KWT_RL_DATA     path of the labelled RL file (default /opt/kwtfs/in/rl.jsonl)
  KWT_UTILITIES   route_regret source: "utilities" (mode-value probe column,
                  falling back to the label) or "label" (fixed measured profile)
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

from kwtfs import format as fmt

FORCED_SHARE = 0.3
DEFAULT_PATH = os.environ.get("KWT_RL_DATA", "/opt/kwtfs/in/rl.jsonl")


def _unit(text: str) -> float:
  return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def control_for(item_id: str, options: list[str]) -> str:
  """Deterministic per item: 70% auto, 30% split over the feasible modes."""
  if _unit("ctl" + item_id) >= FORCED_SHARE:
    return "auto"
  modes = ["jev", "direct", "cot"] if options else ["direct", "cot"]
  return modes[int(_unit("mode" + item_id) * len(modes)) % len(modes)]


def row(rec: dict, utilities_key: str = "utilities") -> dict:
  options = rec["options"]
  control = control_for(rec["id"], options)
  # "label" = the fixed measured profile: its utilities live under label["u"].
  # (A first E6 run passed the whole label dict here, so route_regret found no
  # utility for any mode and became a constant that GRPO normalises away.)
  utilities = (rec.get("label") or {}).get("u") if utilities_key == "label" else (
      rec.get(utilities_key) or (rec.get("label") or {}).get("u"))
  return {
      "id": rec["id"],
      "prompts": fmt.chat_prompt(fmt.render_question(rec["question"], options), control),
      "answer": str(rec["gold"]),
      "options": json.dumps(options),
      "control": control,
      "genre": rec["genre"],
      "utilities": json.dumps(utilities or {}),
  }


def load(path: str | pathlib.Path = DEFAULT_PATH, limit: int | None = None,
         utilities_key: str = "utilities") -> list[dict]:
  with open(path, encoding="utf-8") as handle:
    rows = [row(json.loads(line), utilities_key) for line in handle if line.strip()]
  rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())  # mix genres
  return rows[:limit] if limit else rows


def create_dataset(path: str = DEFAULT_PATH, limit: int | None = None):
  """Tunix CLI `data_module` entry point."""
  import grain  # pylint: disable=g-import-not-at-top

  key = os.environ.get("KWT_UTILITIES", "utilities")  # "label" -> fixed profile (E6)
  return grain.MapDataset.source(load(path, limit, key))
