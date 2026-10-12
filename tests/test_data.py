import collections
import pathlib
import tempfile

import pytest

from kwtfs.data import dedup, schema, sources
from kwtfs.data.schema import Item


def _item(**kw):
  base = dict(id="x", source="s", licence="l", genre="arith", domain="d",
              question="What is 2+2?", options=["3", "4", "5"], gold="4",
              answer_type="number")
  return Item(**{**base, **kw})


def test_schema_flags_bad_options():
  assert schema.problems(_item()) == []
  assert any("2 times" in p for p in schema.problems(_item(options=["4", "4 ", "5"])))
  assert any("0 times" in p for p in schema.problems(_item(options=["3", "5"])))
  assert any("duplicate" in p for p in schema.problems(_item(options=["3", "3", "4"])))
  assert any("non-numeric" in p for p in schema.problems(_item(gold="four", options=[])))
  assert any(">" in p for p in schema.problems(_item(options=list("abcdefg4"), gold="4")))


def test_schema_roundtrip():
  it = _item(difficulty={"digits": 1})
  with tempfile.TemporaryDirectory() as tmp:  # not tmp_path: some Windows setups deny pytest's basetemp
    path = pathlib.Path(tmp) / "a.jsonl"
    schema.write(path, [it])
    back = schema.read(path)[0]
  assert back == it and back.provenance_hash == it.provenance_hash


def test_dedup_keeps_same_text_with_different_answer():
  a = _item(id="a", question="Ann has 3 apples and buys 4 more apples at the shop today. How many?")
  b = _item(id="b", question="Ann has 3 apples and buys 4 more apples at the shop today. How many ?")
  c = _item(id="c", question=a.question, gold="5", options=["3", "5", "6"])
  kept, dropped = dedup.dedup([a, b, c])
  assert [k.id for k in kept] == ["a", "c"] and dropped == {"b": "a"}


def test_procedural_is_valid_and_balanced():
  items = sources.procedural(n_per_cell=20)
  assert all(not schema.problems(it) for it in items)
  compare = [it for it in items if it.source == "proc-compare"]
  gold = collections.Counter(it.gold for it in compare)
  assert abs(gold["yes"] - gold["no"]) <= len(compare) * 0.05
  first = collections.Counter(it.options[0] for it in compare)
  assert min(first.values()) > len(compare) * 0.35  # options are shuffled
  for it in items:  # procedural gold is re-derivable from the question
    if it.source == "proc-compare":
      a, b = [int(w) for w in it.question.split() if w.isdigit()]
      assert it.gold == ("yes" if a > b else "no")
  scores = collections.Counter(it.gold for it in items if it.domain == "rubric-depth2")
  assert len(scores) == 6 and max(scores.values()) - min(scores.values()) <= 1


def test_profile_grading_is_exact():
  from kwtfs.data import profile
  mcq = _item(options=["3", "4", "5"], gold="4")
  assert profile.grade(mcq, "B") and profile.grade(mcq, "(B)") and profile.grade(mcq, "B) 4")
  assert profile.grade(mcq, "4") and not profile.grade(mcq, "C") and not profile.grade(mcq, "Because")
  num = _item(options=[], gold="1000")
  assert profile.grade(num, "1,000 pens") and profile.grade(num, "The total is 1000.")
  assert not profile.grade(num, "100") and not profile.grade(num, None)
  cot = "2*500=1000\n<answer>1000</answer>"
  assert profile.extract(cot, "cot") == "1000"
  assert profile.extract("so the answer is 7.", "cot") == "7"
  assert profile.extract("no final line", "cot") is None
  assert profile.modes_for(mcq) == ("jev", "direct", "cot") and profile.modes_for(num) == ("direct", "cot")


def test_profile_prompts_match_qwen_template():
  from kwtfs.data import profile
  mcq = _item()
  assert profile.prompt(mcq, "jev").endswith("</think>\n\nAnswer:\n")
  assert "A) 3" in profile.prompt(mcq, "cot")
  assert not profile.prompt(mcq, "think").endswith("</think>\n\n")  # thinking left on


def test_audit_set_covers_eval_splits():
  from kwtfs.data import profile
  items = [_item(id=f"{s}{i}", split=s, source="aqua_rat") for s in ("val", "test", "sft") for i in range(3)]
  chosen = profile.audit_set(items, per_source=2)
  assert {it.split for it in chosen} == {"val", "test", "sft"} and len(chosen) == 8


def test_labels_pick_cheapest_mode_that_works():
  from kwtfs.data import labels
  easy = {"jev": {"correct": [True], "tokens": [1]},
          "direct": {"correct": [True] * 4, "tokens": [8] * 4},
          "cot": {"correct": [True] * 4, "tokens": [120] * 4}}
  hard = {"direct": {"correct": [False, False, True, False], "tokens": [8] * 4},
          "cot": {"correct": [True] * 4, "tokens": [150] * 4}}
  assert labels.label(labels.utilities(easy))["suggested"] == "jev"
  lab = labels.label(labels.utilities(hard))
  assert lab["suggested"] == "cot" and not lab["ambiguous"]
  tie = {"direct": {"correct": [True] * 4, "tokens": [8] * 4},
         "cot": {"correct": [True] * 4, "tokens": [40] * 4}}
  assert labels.label(labels.utilities(tie))["ambiguous"]


def test_label_report_gates():
  from kwtfs.data import labels
  items, profiles = [], {}
  for i in range(40):  # same genre; difficulty decides the mode
    hard = i % 2 == 0
    it = _item(id=f"a{i}", options=[], difficulty={"digits": 5 if hard else 1})
    items.append(it)
    profiles[it.id] = {
        "direct": {"correct": [not hard] * 4, "tokens": [8] * 4},
        "cot": {"correct": [True] * 4, "tokens": [300] * 4}}
  rep = labels.report(items, profiles)
  assert rep["gates"]["within_genre_variance"]["arith"]
  assert rep["gates"]["headroom_ok"] and rep["gates"]["stability"] == 1.0


def test_mode_value_probe_learns_a_separable_signal():
  pytest.importorskip("sklearn")
  import numpy as np
  from kwtfs import mode_value
  rng = np.random.default_rng(0)
  n, d = 400, 32
  H = rng.normal(size=(n, d)).astype(np.float16)
  hard = H[:, 0] > 0  # feature 0 says "needs reasoning"
  ids = [f"i{k}" for k in range(n)]
  profiles = {x: {"direct": {"correct": [not h] * 4, "tokens": [8] * 4},
                  "cot": {"correct": [True] * 4, "tokens": [120] * 4}}
              for x, h in zip(ids, hard)}
  items = {x: _item(id=x, options=[]) for x in ids}
  model = mode_value.fit(H, ids, profiles, set(range(300)))
  rep = mode_value.evaluate(model, H, ids, profiles, set(range(300, n)), items)
  assert rep["direct"]["auroc"] > 0.95
  assert rep["router_utility"]["probe"] > rep["router_utility"]["oracle"] - 0.02


def test_sft_builder_obey_and_route():
  from kwtfs.data import sft_data
  mcq = _item(id="m", genre="arith", options=["3", "4", "5"], gold="4")
  hard = _item(id="h", genre="arith", options=[], gold="35",
               question="4 cows and 3 sheep eat 2 each, 7 chickens eat 3. Total?")
  profiles = {
      "m": {"jev": {"correct": [True], "tokens": [1]},
            "direct": {"correct": [True] * 4, "tokens": [3] * 4},
            "cot": {"correct": [True] * 4, "tokens": [60] * 4}},
      "h": {"direct": {"correct": [False] * 4, "tokens": [3] * 4},
            "cot": {"correct": [True] * 4, "tokens": [60] * 4}}}
  traces = {"h": [
      {"text": "4*2=8\n3*2=7\n7*3=21\n<answer>35</answer>", "tokens": 20, "correct": True},  # false line
      {"text": "4*2=8\n3*2=6\n7*3=21\n8+6+21=35\n<answer>35</answer>", "tokens": 30, "correct": True}]}
  data, stats = sft_data.build([mcq, hard], profiles, traces, obedience_share=0.5)
  route = {ex["id"]: ex for ex in data if ex["kind"] == "route"}
  assert route["m"]["mode"] == "jev" and route["h"]["mode"] == "cot"
  assert "3*2=7" not in route["h"]["completion"]  # unfaithful trace rejected
  assert route["h"]["completion"].startswith("<genre:arith><mode:cot>\n4*2=8")
  assert route["h"]["completion"].endswith("<answer>35</answer><|im_end|>")
  assert "Mode: auto." in route["h"]["prompt"] and stats["route"] == 2
