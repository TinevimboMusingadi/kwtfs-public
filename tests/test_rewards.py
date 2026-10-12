"""Exploit checks: no cheap path beats honest work."""
import json

from kwtfs import rewards as R

TERMS = (R.correctness, R.well_formed, R.compute_cost, R.faithful, R.genre, R.route_regret)


def total(comp, answer, options, control="auto", genre="arith", utilities=None):
  kw = dict(answer=[answer], options=[json.dumps(options)], control=[control],
            genre=[genre], utilities=[json.dumps(utilities) if utilities else None])
  return sum(f(None, [comp], **kw)[0] for f in TERMS)


def test_short_wrong_never_beats_long_right():
  long_right = "<genre:arith><mode:cot>\n" + "4682*2=9364\n" * 20 + "<answer>9364</answer>"
  short_wrong = "<genre:arith><mode:direct>\n<answer>9000</answer>"
  assert total(long_right, "9364", []) > total(short_wrong, "9364", [])


def test_false_scratch_then_lucky_pick_scores_below_clean_trace():
  opts = ["52020175", "52020176", "52020215", "52020235"]
  lucky = "<genre:arith><mode:cot>\n(7212*7213)+59=79981\n<mode:jev>\nC"
  clean = "<genre:arith><mode:cot>\n(7212*7213)+59=52020215\n<mode:jev>\nC"
  gold = opts[2]
  assert R.faithful(None, [lucky])[0] < 0 and R.faithful(None, [clean])[0] == 0
  assert total(lucky, gold, opts) < total(clean, gold, opts)


def test_disobeying_a_forced_mode_is_worst():
  opts = ["4", "5"]
  disobey_right = "<genre:arith><mode:jev>\nB"
  obey_wrong = "<genre:arith><mode:cot>\n2+2=4\n<mode:jev>\nA"
  assert total(disobey_right, "5", opts, control="cot") < total(obey_wrong, "5", opts, control="cot")


def test_route_regret_is_zero_for_best_or_tied_mode():
  u = {"jev": 0.9, "direct": 0.88, "cot": 0.5}
  kw = dict(options=[json.dumps(["a", "b"])], control=["auto"], utilities=[json.dumps(u)])
  jev = R.route_regret(None, ["<mode:jev>\nA"], **kw)[0]
  direct = R.route_regret(None, ["<mode:direct>\n<answer>a</answer>"], **kw)[0]
  cot = R.route_regret(None, ["<mode:cot>\nx\n<answer>a</answer>"], **kw)[0]
  assert jev == 0 and -0.02 < direct < 0 and cot < direct
  forced = R.route_regret(None, ["<mode:cot>\nx\n<answer>a</answer>"],
                          options=kw["options"], control=["cot"], utilities=kw["utilities"])[0]
  assert forced == 0  # no routing penalty when the user chose the mode


def test_wrong_answers_pay_no_token_cost():
  wrong = "<mode:cot>\n" + "x" * 400 + "\n<answer>1</answer>"
  assert R.compute_cost(None, [wrong], answer=["2"], options=["[]"], control=["auto"])[0] == 0


def test_rl_rows_mix_controls_and_carry_reward_columns():
  import collections
  from kwtfs import format as fmt
  from kwtfs import tasks
  recs = [{"id": f"x{i}", "question": "2+2?", "options": ["3", "4"] if i % 2 else [],
           "gold": "4", "genre": "arith", "label": {"u": {"direct": 0.9, "cot": 0.8}}}
          for i in range(2000)]
  rows = [tasks.row(r) for r in recs]
  share = collections.Counter(r["control"] for r in rows)
  assert 0.65 < share["auto"] / len(rows) < 0.75
  assert all(r["control"] != "jev" for r in rows if r["options"] == "[]")
  r = rows[0]
  assert f"Mode: {r['control']}." in r["prompts"] and json.loads(r["utilities"])["direct"] == 0.9
  assert r["prompts"] == fmt.chat_prompt(fmt.render_question("2+2?", json.loads(r["options"])),
                                            r["control"])


def test_eval_scoring():
  from kwtfs.eval import evaluate as E
  lab = {"suggested": "jev", "ambiguous": False, "u": {"jev": 0.99, "direct": 0.9, "cot": 0.8}}
  rows = [{"question": "q", "options": ["4", "5"], "gold": "5", "genre": "arith", "label": lab},
          {"question": "q", "options": [], "gold": "7", "genre": "arith", "label": None}]
  res = [{"completion": "<genre:arith><mode:jev>\nB", "tokens": 6},
         {"completion": "<genre:arith><mode:cot>\n3+4=8\n<answer>7</answer>", "tokens": 20}]
  rep = E.score(rows, res, "auto")["all"]
  assert rep["accuracy"] == 1.0 and rep["routing_acc"] == 1.0 and rep["regret"] == 0
  assert rep["faithful"] == 0.0 and rep["genre_tag_acc"] == 1.0
  forced = E.score(rows, res, "cot")["all"]
  assert forced["obedience"] == 0.5 and forced["accuracy"] == 0.5
  assert E._HEAD.match("<genre:arith><mode:cot>\nabc").group(2) == "cot"


def test_route_regret_anneals_to_zero():
  import importlib
  r = importlib.reload(R)
  r._seen = 0
  r.ANNEAL_COMPLETIONS = 4
  kw = dict(options=[json.dumps(["a", "b"])], control=["auto"],
            utilities=[json.dumps({"jev": 1.0, "cot": 0.5})])
  vals = [r.route_regret(None, ["<mode:cot>\nx\n<answer>a</answer>"], **kw)[0] for _ in range(6)]
  assert vals[0] == -0.25 and vals[2] > vals[0] and vals[4] == 0 and vals[5] == 0


def test_eval_run_assembles_handoff_traces():
  """An early eval scored every CoT->JEV hand-off as invalid: the scored string dropped
  the <mode:jev> tag. Drive run() with a fake model to pin the assembly."""
  from kwtfs.eval import evaluate as E

  class Fake(E.Runner):
    def __init__(self):  # no model
      self.tok = type("T", (), {"encode": staticmethod(lambda s: s.split())})()

    def generate(self, prompts, steps):
      if steps == E.HEAD_STEPS:
        return ["<genre:compare><mode:cot>\n"] * len(prompts), [5] * len(prompts)
      return ["5 > 4.\n<mode:jev>\nB"] * len(prompts), [6] * len(prompts)

    def jev(self, prefixes, num_options):
      assert all(p.endswith("<mode:jev>\n") for p in prefixes)
      return [{"A": 0.1, "B": 0.9}] * len(prefixes)

  rows = [{"question": "Is 5 > 4?", "options": ["no", "yes"], "gold": "yes", "genre": "compare"}]
  res = Fake().run(rows, "cot")
  assert res[0]["completion"] == "<genre:compare><mode:cot>\n5 > 4.\n<mode:jev>\nB"
  assert E.score(rows, res, "cot")["all"]["accuracy"] == 1.0


def test_rewards_accept_numpy_columns():
  import numpy as np
  comps = ["<genre:arith><mode:jev>\nB"]
  kw = dict(answer=np.array(["5"]), options=np.array([json.dumps(["4", "5"])]),
            control=np.array(["auto"]), genre=np.array(["arith"]),
            utilities=np.array([json.dumps({"jev": 1.0, "direct": 0.9})]))
  assert sum(f(None, comps, **kw)[0] for f in TERMS) > 1.0


def test_fixed_profile_utilities_column():
  from kwtfs import tasks
  rec = {"id": "x", "question": "q", "options": [], "gold": "1", "genre": "arith",
         "utilities": {"direct": 0.1, "cot": 0.2},
         "label": {"suggested": "cot", "u": {"direct": 0.5, "cot": 0.9}}}
  assert json.loads(tasks.row(rec, "label")["utilities"]) == {"direct": 0.5, "cot": 0.9}
  assert json.loads(tasks.row(rec)["utilities"]) == {"direct": 0.1, "cot": 0.2}
