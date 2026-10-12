"""Verify the arithmetic written in reasoning traces.

An earlier (1.7B) run produced correct JEV picks after false scratch work such as
`(7212*7213)+59=79981`. These checks find `expression = number` claims in a
trace and evaluate the left side safely (numbers and + - * / // % ** only).
Used to filter self-distilled SFT traces and by the `faithful` reward.
"""

from __future__ import annotations

import ast
import dataclasses
import operator
import re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
# `lhs = rhs` where lhs has at least one operator; rhs is a plain number.
_CLAIM = re.compile(
    r"(?P<lhs>[\d\s.,()+\-*/×x÷%^]*\d[\d\s.,()]*[+\-*/×x÷%^][\d\s.,()+\-*/×x÷%^]*\d\)?)\s*=\s*"
    r"\$?(?P<rhs>-?\d[\d,]*(?:\.\d+)?)")
_MAX_POW = 64


@dataclasses.dataclass(frozen=True)
class Claim:
  text: str
  claimed: float
  actual: float | None  # None: could not evaluate (not counted either way)

  @property
  def true(self) -> bool | None:
    if self.actual is None:
      return None
    return abs(self.actual - self.claimed) <= 1e-6 * max(1.0, abs(self.actual)) + 0.005


def _eval(node):
  if isinstance(node, ast.Expression):
    return _eval(node.body)
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    return node.value
  if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
    return _OPS[type(node.op)](_eval(node.operand))
  if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
    left, right = _eval(node.left), _eval(node.right)
    if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
      raise ValueError("exponent too large")
    return _OPS[type(node.op)](left, right)
  raise ValueError(f"unsupported: {ast.dump(node)[:40]}")


def evaluate(expr: str) -> float | None:
  expr = (expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
          .strip())
  expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expr)  # 3 x 4
  try:
    return float(_eval(ast.parse(expr, mode="eval")))
  except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
    return None


def claims(trace: str) -> list[Claim]:
  out = []
  for line in trace.splitlines():
    for m in _CLAIM.finditer(line):
      lhs = m.group("lhs").strip()
      # Chained "a+b=c+d=e": only the segment right before this "=" counts.
      lhs = re.split(r"=", lhs)[-1]
      rhs = float(m.group("rhs").replace(",", ""))
      out.append(Claim(f"{lhs}={m.group('rhs')}", rhs, evaluate(lhs)))
  return out


def false_claims(trace: str) -> list[Claim]:
  return [c for c in claims(trace) if c.true is False]


def faithful(trace: str) -> bool:
  """True when no checkable equation in the trace is wrong."""
  return not false_claims(trace)
