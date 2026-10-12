"""Mode-value model: predict p(correct | question, mode) and tokens per mode.

A calibrated alternative to "a small reward model overfit on SFT data": it
never judges correctness (that stays verifiable). It predicts how well each
mode would do on a question, which gives
  * the `utilities` column used by the route_regret reward on RL prompts,
  * a probe router baseline for the paper, and
  * the answer to "does the model already know when it needs to think?"

Features are Qwen3-4B's final hidden state at the last token of the auto
prompt, i.e. exactly where the model will decide its mode.

    # TPU (one forward pass per item, ~minutes):
    python -m kwtfs.mode_value extract --items in/items.jsonl --out out
    # local CPU:
    python -m kwtfs.mode_value train --hidden runs/.../hidden.npz \
        --profile runs/.../profile.jsonl --out runs/mode_value
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np

from kwtfs import format as fmt
from kwtfs.data import labels, schema

MODES = labels.MODES


# -- TPU: hidden states --------------------------------------------------------

def extract(items: list[schema.Item], model, tokenizer, batch: int = 32) -> np.ndarray:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from tunix.generate import utils as gen_utils  # pylint: disable=g-import-not-at-top

  out = []
  pad = tokenizer.pad_id()
  for s in range(0, len(items), batch):
    chunk = items[s:s + batch]
    ids = [tokenizer.encode(fmt.chat_prompt(fmt.render_question(it.question, it.options)))
           for it in chunk]
    ids += [ids[0]] * (batch - len(ids))  # fixed shape
    length = -(-max(len(i) for i in ids) // 256) * 256
    tokens = jnp.asarray([[pad] * (length - len(i)) + i for i in ids], jnp.int32)
    mask = jnp.asarray([[0] * (length - len(i)) + [1] * len(i) for i in ids], jnp.bool_)
    hidden, _ = model(tokens, gen_utils.build_positions_from_mask(mask), None,
                      gen_utils.make_causal_attn_mask(mask, length), skip_lm_head=True)
    last = np.asarray(jax.device_get(hidden[:, -1, :]), np.float32)
    out.append(last[:len(chunk)])
    print(f"[mode_value] hidden {s + len(chunk)}/{len(items)}", flush=True)
  return np.concatenate(out).astype(np.float16)


# -- local: probes -------------------------------------------------------------

class _Constant:
  """Stands in for a probe when a mode is always (or never) right in training."""

  def __init__(self, p: float):
    self.p = p

  def predict_proba(self, Z):
    return np.tile([1 - self.p, self.p], (len(Z), 1))


def _targets(profiles: dict, ids: list[str]):
  """Per mode: (row index, #correct, #samples, mean tokens) for items that have it."""
  out = {}
  for m in MODES:
    rows = [(i, sum(profiles[x][m]["correct"]), len(profiles[x][m]["correct"]),
             float(np.mean(profiles[x][m]["tokens"])))
            for i, x in enumerate(ids) if m in profiles.get(x, {})]
    out[m] = rows
  return out


def fit(H: np.ndarray, ids: list[str], profiles: dict, train_idx: set[int],
        Cs=(1e-3, 1e-2, 1e-1, 1.0)):
  """One linear probe per mode on the full standardised hidden state (K binary
  outcomes per item, so p_m is learned as a probability) + ridge on
  log-tokens. The L2 strength is chosen by 3-fold CV on log-loss: the point is
  calibration on unseen questions, not memorising the profile. (No PCA: on a
  synthetic test the "needs reasoning" direction was not a top-variance one.)"""
  from sklearn.linear_model import LogisticRegressionCV, Ridge  # pylint: disable=g-import-not-at-top
  from sklearn.preprocessing import StandardScaler  # pylint: disable=g-import-not-at-top

  tr = sorted(train_idx)
  scaler = StandardScaler().fit(H[tr].astype(np.float32))
  Z = scaler.transform(H.astype(np.float32))
  probes = {}
  for m, rows in _targets(profiles, ids).items():
    rows = [r for r in rows if r[0] in train_idx]
    if len(rows) < 50:
      continue
    X, y, w = [], [], []
    for i, c, n, _ in rows:  # each item contributes its K outcomes as weights
      if c:
        X.append(Z[i]); y.append(1); w.append(c / n)
      if n - c:
        X.append(Z[i]); y.append(0); w.append((n - c) / n)
    if len(set(y)) == 1:
      clf = _Constant(float(y[0]))
    else:
      clf = LogisticRegressionCV(Cs=list(Cs), cv=3, scoring="neg_log_loss",
                                 max_iter=2000).fit(np.array(X), y, sample_weight=w)
    reg = Ridge(alpha=10.0).fit(Z[[r[0] for r in rows]], np.log1p([r[3] for r in rows]))
    probes[m] = (clf, reg)
  return {"scaler": scaler, "probes": probes}


def predict(model: dict, H: np.ndarray) -> dict[str, np.ndarray]:
  Z = model["scaler"].transform(H.astype(np.float32))
  return {m: (clf.predict_proba(Z)[:, 1], np.expm1(reg.predict(Z)))
          for m, (clf, reg) in model["probes"].items()}


def evaluate(model, H, ids, profiles, idx, items_by_id, lam=labels.LAMBDA) -> dict:
  from sklearn.metrics import roc_auc_score  # pylint: disable=g-import-not-at-top

  idx = sorted(idx)
  pred = predict(model, H[idx])
  rep = {}
  for m, (p, _) in pred.items():
    rows = [(k, profiles[ids[i]][m]) for k, i in enumerate(idx) if m in profiles.get(ids[i], {})]
    if not rows:
      continue
    true_p = np.array([sum(r["correct"]) / len(r["correct"]) for _, r in rows])
    pp = p[[k for k, _ in rows]]
    y = (true_p >= 0.5).astype(int)
    bins = np.clip((pp * 10).astype(int), 0, 9)
    ece = sum(abs(pp[bins == b].mean() - true_p[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    rep[m] = {"n": len(rows), "auroc": round(float(roc_auc_score(y, pp)), 4)
              if 0 < y.sum() < len(y) else None,
              "brier": round(float(np.mean((pp - true_p) ** 2)), 4), "ece": round(float(ece), 4)}
  # probe router: pick argmax predicted utility, score it with measured utility
  chosen, oracle, lookup_u = [], [], []
  share = {}
  for k, i in enumerate(idx):
    it = items_by_id[ids[i]]
    u_true = labels.utilities(profiles[ids[i]], lam)
    avail = [m for m in pred if m in u_true]
    u_pred = {m: pred[m][0][k] - lam * pred[m][1][k] for m in avail}
    pick = max(u_pred, key=u_pred.get)
    chosen.append(u_true[pick]["u"])
    oracle.append(max(v["u"] for v in u_true.values()))
    share.setdefault(it.genre, []).append(pick)
  rep["router_utility"] = {"probe": round(float(np.mean(chosen)), 4),
                           "oracle": round(float(np.mean(oracle)), 4)}
  rep["probe_mode_share"] = {g: {m: v.count(m) / len(v) for m in set(v)} for g, v in share.items()}
  return rep


def main() -> None:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  ex = sub.add_parser("extract")
  ex.add_argument("--items", default="in/items.jsonl")
  ex.add_argument("--out", required=True)
  ex.add_argument("--model", default="qwen3-4b=Qwen/Qwen3-4B")
  ex.add_argument("overrides", nargs="*")
  tr = sub.add_parser("train")
  tr.add_argument("--items", default="data/kwtfs/items.jsonl")
  tr.add_argument("--hidden", required=True)
  tr.add_argument("--profile", required=True)
  tr.add_argument("--out", required=True)
  args = ap.parse_args()

  if args.cmd == "extract":
    from kwtfs.train import grpo  # pylint: disable=g-import-not-at-top
    name, hf_id = args.model.split("=")
    model, tok = grpo.load_model(None, grpo.model_overrides(name, hf_id)
                                 + ["model_config.dtype=bfloat16"] + args.overrides)
    items = schema.read(args.items)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H = extract(items, model, tok)
    np.savez_compressed(out / "hidden.npz", H=H, ids=np.array([it.id for it in items]))
    print(f"[mode_value] saved {H.shape} -> {out / 'hidden.npz'}", flush=True)
    return

  items = {it.id: it for it in schema.read(args.items)}
  data = np.load(args.hidden)
  H, ids = data["H"], [str(x) for x in data["ids"]]
  profiles = labels.load_profiles(args.profile)
  split = [items[x].split for x in ids]
  train_idx = {i for i, s in enumerate(split) if s in ("sft", "rl")}
  val_idx = {i for i, s in enumerate(split) if s == "val"}
  model = fit(H, ids, profiles, train_idx)
  rep = {"val": evaluate(model, H, ids, profiles, val_idx, items),
         "train": evaluate(model, H, ids, profiles, train_idx, items)}
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  with open(out / "mode_value.pkl", "wb") as handle:
    pickle.dump(model, handle)
  (out / "report.json").write_text(json.dumps(rep, indent=2))
  print(json.dumps(rep, indent=2))


if __name__ == "__main__":
  main()
