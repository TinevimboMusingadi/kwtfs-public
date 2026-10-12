"""Turn profiles into suggested-mode labels, and check the labels are usable.

    python -m kwtfs.data.labels --profile runs/<run>/out/profile.jsonl --report

For item x and mode m (measured on base Qwen3-4B):
  p_m = accuracy over K samples (JEV: argmax correctness, one forward step)
  t_m = mean generated tokens
  U_m = p_m - LAMBDA * t_m        suggested(x) = argmax_m U_m
Items whose best two utilities are within TIE are "ambiguous": they get a soft
label but are left out of routing-accuracy metrics.

Label-quality gates are printed, not enforced:
  within-genre variance  >= 2 modes with >= 15% share in each genre
  headroom               oracle utility - genre-lookup utility >= 0.05
  stability              suggested-label agreement across two halves of K >= 85%
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics

from kwtfs.data import schema

LAMBDA = 0.0005  # 100 CoT tokens cost 5 accuracy points
TIE = 0.05
MODES = ("jev", "direct", "cot")


def load_profiles(path) -> dict[str, dict[str, dict]]:
  """id -> mode -> record (think/audit records kept under mode 'think')."""
  out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
  with open(path, encoding="utf-8") as handle:
    for line in handle:
      r = json.loads(line)
      out[r["id"]][r["mode"]] = r
  return out


def utilities(modes: dict[str, dict], lam: float = LAMBDA, half: int | None = None) -> dict:
  """U_m per mode. half=0/1 uses only the even/odd samples (stability check)."""
  u = {}
  for m in MODES:
    r = modes.get(m)
    if not r:
      continue
    correct, tokens = r["correct"], r["tokens"]
    if half is not None and len(correct) > 1:
      correct, tokens = correct[half::2], tokens[half::2]
    p = sum(correct) / len(correct)
    u[m] = {"p": p, "t": statistics.mean(tokens), "u": p - lam * statistics.mean(tokens)}
  return u


def label(u: dict) -> dict:
  ranked = sorted(u, key=lambda m: u[m]["u"], reverse=True)
  best = ranked[0]
  gap = u[best]["u"] - u[ranked[1]]["u"] if len(ranked) > 1 else 1.0
  return {"suggested": best, "ambiguous": gap < TIE, "gap": gap,
          "u": {m: round(v["u"], 4) for m, v in u.items()},
          "p": {m: round(v["p"], 4) for m, v in u.items()},
          "t": {m: round(v["t"], 1) for m, v in u.items()}}


def difficulty_band(it: schema.Item) -> str:
  d = it.difficulty
  for key in ("digits", "steps", "depth", "gap", "qtype"):
    if key in d:
      return f"{key}={d[key]}"
  return "-"


def report(items: list[schema.Item], profiles: dict, lam: float = LAMBDA) -> dict:
  rows = []
  for it in items:
    modes = profiles.get(it.id, {})
    if not all(m in modes for m in ("direct", "cot")):
      continue
    u = utilities(modes, lam)
    rows.append((it, u, label(u)))
  if not rows:
    return {"n": 0}

  # mode share per genre, and per genre x difficulty band
  share = collections.defaultdict(collections.Counter)
  band = collections.defaultdict(collections.Counter)
  for it, _, lab in rows:
    share[it.genre][lab["suggested"]] += 1
    band[(it.genre, difficulty_band(it))][lab["suggested"]] += 1

  # routers: oracle = per-item best; genre-lookup = the single mode with the
  # best mean utility in each genre (the strongest genre-only router; a
  # majority vote over labels is weaker, e.g. rubric: jev wins 152-146 on
  # labels but always-cot has twice the utility).
  def _u(u, m):
    return u[m]["u"] if m in u else u["direct"]["u"]
  lookup = {}
  for g in share:
    genre_rows = [u for it, u, _ in rows if it.genre == g]
    lookup[g] = max(MODES, key=lambda m: sum(_u(u, m) for u in genre_rows))
  util = collections.defaultdict(lambda: collections.defaultdict(list))
  for it, u, lab in rows:
    g = it.genre
    util[g]["oracle"].append(u[lab["suggested"]]["u"])
    util[g]["lookup"].append(u[lookup[g]]["u"] if lookup[g] in u else u["cot"]["u"])
    for m in MODES:
      util[g][f"always-{m}"].append(u[m]["u"] if m in u else u["direct"]["u"])
  routers = {g: {k: round(statistics.mean(v), 4) for k, v in d.items()} for g, d in util.items()}
  allr = collections.defaultdict(list)
  for it, u, lab in rows:
    allr["oracle"].append(u[lab["suggested"]]["u"])
    allr["lookup"].append(u[lookup[it.genre]]["u"] if lookup[it.genre] in u else u["cot"]["u"])
    for m in MODES:
      allr[f"always-{m}"].append(u[m]["u"] if m in u else u["direct"]["u"])
  # Cross-fitted oracle: pick the mode on even samples, score it on odd ones
  # (and vice versa). The plain oracle maxes over noisy K=4 estimates and so
  # overstates what any router can reach (winner's curse).
  for it, u, lab in rows:
    halves = [utilities(profiles[it.id], lam, half=h) for h in (0, 1)]
    picks = [max(h, key=lambda m: h[m]["u"]) for h in halves]
    allr["oracle_xfit"].append(statistics.mean(
        [halves[1][picks[0]]["u"], halves[0][picks[1]]["u"]]))
    allr["lookup_xfit"].append(statistics.mean(
        [_u(halves[1], lookup[it.genre]), _u(halves[0], lookup[it.genre])]))
  routers["ALL"] = {k: round(statistics.mean(v), 4) for k, v in allr.items()}

  # stability: labels from even vs odd samples (needs K >= 2)
  agree = n_stab = 0
  for it, _, lab in rows:
    if lab["ambiguous"]:
      continue
    a = label(utilities(profiles[it.id], lam, half=0))["suggested"]
    b = label(utilities(profiles[it.id], lam, half=1))["suggested"]
    agree += a == b
    n_stab += 1

  variance_ok = {g: sum(v / sum(c.values()) >= 0.15 for v in c.values()) >= 2
                 for g, c in share.items()}
  headroom = routers["ALL"]["oracle_xfit"] - routers["ALL"]["lookup_xfit"]
  return {
      "n": len(rows), "lambda": lam,
      "ambiguous": sum(lab["ambiguous"] for _, _, lab in rows) / len(rows),
      "share_by_genre": {g: dict(c) for g, c in share.items()},
      "share_by_band": {f"{g} | {b}": dict(c) for (g, b), c in sorted(band.items())},
      "mean_p": {m: round(statistics.mean(u[m]["p"] for _, u, _ in rows if m in u), 4)
                 for m in MODES if any(m in u for _, u, _ in rows)},
      "mean_tokens": {m: round(statistics.mean(u[m]["t"] for _, u, _ in rows if m in u), 1)
                      for m in MODES if any(m in u for _, u, _ in rows)},
      "routers_utility": routers,
      "gates": {
          "within_genre_variance": variance_ok,
          "headroom": round(headroom, 4), "headroom_ok": headroom >= 0.05,
          "stability": round(agree / max(1, n_stab), 4),
          "stability_ok": agree / max(1, n_stab) >= 0.85,
      },
  }


def audit_queue(items: list[schema.Item], profiles: dict, conf: float = 0.9) -> list[dict]:
  """Items where native thinking confidently disagrees with gold -> review."""
  queue = []
  for it in items:
    r = profiles.get(it.id, {}).get("think")
    if not r or r.get("correct", [True])[0]:
      continue
    c = r.get("confidence")
    if c is None or c >= conf:  # open-ended answers have no confidence: review all
      queue.append({"id": it.id, "source": it.source, "split": it.split,
                    "question": it.question, "options": it.options, "gold": it.gold,
                    "model_answer": r["answers"][0], "confidence": c,
                    "finished_thinking": r.get("finished"), "thinking_tail": r.get("sample")})
  return queue


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--items", default="data/kwtfs/items.jsonl")
  ap.add_argument("--profile", required=True)
  ap.add_argument("--lambda", dest="lam", type=float, default=LAMBDA)
  ap.add_argument("--report", action="store_true")
  ap.add_argument("--write", help="write items + labels jsonl here")
  ap.add_argument("--queue", help="write the audit review queue here")
  args = ap.parse_args()
  items = schema.read(args.items)
  profiles = load_profiles(args.profile)
  if args.report:
    print(json.dumps(report(items, profiles, args.lam), indent=2))
  if args.queue:
    q = audit_queue(items, profiles)
    with open(args.queue, "w", encoding="utf-8") as handle:
      for r in q:
        handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    audited = collections.Counter(it.source for it in items if "think" in profiles.get(it.id, {}))
    flagged = collections.Counter(r["source"] for r in q)
    print("audit flagged / audited:", {s: f"{flagged[s]}/{n}" for s, n in audited.items()})
  if args.write:
    with open(args.write, "w", encoding="utf-8") as handle:
      for it in items:
        modes = profiles.get(it.id, {})
        if all(m in modes for m in ("direct", "cot")):
          rec = json.loads(it.to_json())
          rec["label"] = label(utilities(modes, args.lam))
          handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
  main()
