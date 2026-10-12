"""Trace format shared by data, rewards, runtime and evaluation.

The system prompt carries a mode control, ``Mode: auto|jev|direct|cot``. A
completion starts with a genre tag and then a sequence of mode segments:

    <genre:arith><mode:direct>\n<answer>42</answer>
    <genre:arith><mode:cot>\n...reasoning...\n<answer>42</answer>
    <genre:compare><mode:jev>\nB
    <genre:compare><mode:cot>\n...reasoning...\n<mode:jev>\nB

A JEV segment is a single option letter: one categorical token, which is what
lets stock token-level GRPO train the decision. The newline before the letter
keeps the letter a standalone token under Qwen's pre-tokenizer.

Under a forced control the completion must start with that mode (the genre tag
is still emitted). Under ``auto`` the model picks. Genre tags are checked
against the item's genre by the reward; they are not needed for a valid trace.
"""

from __future__ import annotations

import dataclasses
import re

MODES = ("jev", "direct", "cot")
LETTERS = "ABCDEF"
JEV_PREFIX = "<mode:jev>\n"
GENRES = ("decision", "compare", "rubric", "arith", "wordproblem", "knowledge")
CONTROLS = ("auto",) + MODES

SYSTEM_PROMPT = (
    "Start with a genre tag, then exactly one mode tag.\n"
    "Genres: " + ", ".join(f"<genre:{g}>" for g in GENRES) + ".\n"
    "<mode:direct> answer immediately: <mode:direct>\n<answer>VALUE</answer>\n"
    "<mode:cot> think step by step in a few short lines, then end with "
    "<answer>VALUE</answer> or, when options are given, switch to <mode:jev>.\n"
    "<mode:jev> pick one option letter only: <mode:jev>\nB (only when options are given).\n"
    "Mode: {control}. If a mode is named, use it; if auto, choose the cheapest mode "
    "that will be correct."
)

_MODE_RE = re.compile(r"<mode:(jev|direct|cot)>")
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_JEV_RE = re.compile(r"<mode:jev>\s*([A-F])\s*$")
_GENRE_RE = re.compile(r"^\s*<genre:(\w+)>")
_LETTER_ANSWER_RE = re.compile(r"^\s*\(?([A-F])\)?(?=[\s.):,]|$)")


def render_question(prompt: str, options: list[str]) -> str:
  if not options:
    return prompt
  lines = [prompt, "", "Options:"]
  lines += [f"{LETTERS[i]}) {text}" for i, text in enumerate(options)]
  return "\n".join(lines)


def chat_prompt(question: str, control: str = "auto") -> str:
  """Qwen3 chat template with thinking disabled (matches enable_thinking=False)."""
  if control not in CONTROLS:
    raise ValueError(control)
  return (
      f"<|im_start|>system\n{SYSTEM_PROMPT.format(control=control)}<|im_end|>\n"
      f"<|im_start|>user\n{question}<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n"
  )


# -- mode trace (everything after the genre tag) ------------------------------

@dataclasses.dataclass(frozen=True)
class Parsed:
  modes: tuple[str, ...]  # mode tags in order of appearance
  answer: str | None  # option letter (JEV) or free-text value
  via_jev: bool
  valid: bool

  @property
  def first_mode(self) -> str | None:
    return self.modes[0] if self.modes else None

  @property
  def switched(self) -> bool:
    return len(set(self.modes)) > 1


def parse(completion: str, num_options: int) -> Parsed:
  """Strictly parse a mode trace (no genre tag). Anything off-format is invalid."""
  text = completion.strip()
  modes = tuple(_MODE_RE.findall(text))
  if not modes or not text.startswith("<mode:"):
    return Parsed(modes, None, False, False)
  if modes[-1] == "jev":
    m = _JEV_RE.search(text)
    if m is None or num_options == 0 or LETTERS.index(m.group(1)) >= num_options:
      return Parsed(modes, None, True, False)
    # Only cot may precede a JEV decision, and at most once.
    if modes[:-1] not in ((), ("cot",)):
      return Parsed(modes, None, True, False)
    return Parsed(modes, m.group(1), True, True)
  answers = _ANSWER_RE.findall(text)
  if len(modes) != 1 or len(answers) != 1 or not text.endswith("</answer>"):
    return Parsed(modes, None, False, False)
  if modes[0] == "direct" and len(text) > 120:
    return Parsed(modes, None, False, False)  # direct must not hide reasoning
  return Parsed(modes, answers[0], False, True)


def _number(value: str) -> float | None:
  try:
    return float(value.replace(",", "").strip())
  except ValueError:
    return None


def is_correct_exact(parsed: Parsed, gold: str, options: list[str]) -> bool:
  """Exact match: a JEV letter names its option; numbers compare as numbers;
  text compares case-insensitively. No substring credit."""
  if not parsed.valid or parsed.answer is None:
    return False
  value = parsed.answer
  if parsed.via_jev:
    value = options[LETTERS.index(value)]
  a, b = _number(value), _number(gold)
  if a is not None and b is not None:
    return abs(a - b) < 1e-9
  return value.strip().lower() == gold.strip().lower()


def is_correct(parsed: Parsed, gold: str, options: list[str]) -> bool:
  """`is_correct_exact`, plus: with options shown, an <answer> holding just a
  letter (or "B) text") names that option. The capability profile graded
  letters this way (kwtfs/data/profile.py), so rewards and eval must too."""
  if parsed.valid and not parsed.via_jev and options and parsed.answer is not None:
    m = _LETTER_ANSWER_RE.match(parsed.answer)
    if m and LETTERS.index(m.group(1)) < len(options):
      value = options[LETTERS.index(m.group(1))]
      return value.strip().lower() == gold.strip().lower()
  return is_correct_exact(parsed, gold, options)


# -- full completion: genre tag + trace + control ------------------------------

@dataclasses.dataclass(frozen=True)
class Tagged:
  genre: str | None
  trace: Parsed
  obeyed: bool  # first mode matches a forced control (always True under auto)

  @property
  def valid(self) -> bool:
    return self.trace.valid and self.obeyed


def parse_tagged(completion: str, num_options: int, control: str = "auto") -> Tagged:
  m = _GENRE_RE.match(completion)
  genre = m.group(1) if m and m.group(1) in GENRES else None
  rest = completion[m.end():] if m else completion
  trace = parse(rest, num_options)
  obeyed = control == "auto" or trace.first_mode == control
  return Tagged(genre, trace, obeyed)
