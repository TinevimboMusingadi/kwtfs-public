"""CPU-only checks of the training wiring (no JAX / Tunix needed)."""
import importlib
import json

import pytest

from kwtfs.train import grpo

yaml = pytest.importorskip("yaml")


def test_default_config_points_at_this_package():
  path = grpo.config_path()
  assert path.name == "grpo_4b.yaml" and path.is_file()
  cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
  assert cfg["data_module"] == "kwtfs.tasks"
  assert cfg["reward_functions"] == ["kwtfs.rewards"]
  for module in [cfg["data_module"], *cfg["reward_functions"]]:
    importlib.import_module(module)
  assert cfg["model_config"]["model_id"] == "Qwen/Qwen3-4B"
  assert cfg["rollout_config"]["total_generation_steps"] == 256  # = profiling CoT cap


def test_step_metrics_logs_routing(capsys):
  comps = ["<genre:arith><mode:jev>\nB", "<genre:arith><mode:cot>\n1+1=2\n<answer>2</answer>"]
  grpo.step_metrics(None, comps, [1.0, 0.0], None, options=[json.dumps(["1", "2"])] * 2,
                    control=["auto", "cot"], answer=["2", "2"])
  row = json.loads(capsys.readouterr().out.split("[step] ", 1)[1])
  assert row["valid"] == 1.0 and row["correct"] == 1.0
  assert row["mode_jev"] == 0.5 and row["mode_cot"] == 0.5 and row["switched"] == 0.0


def test_model_overrides():
  assert grpo.model_overrides("qwen3-0.6b", "Qwen/Qwen3-0.6B") == [
      "model_config.model_name=qwen3-0.6b", "model_config.model_id=Qwen/Qwen3-0.6B",
      "tokenizer_config.tokenizer_path=Qwen/Qwen3-0.6B"]


def test_tpu_runtime_imports_when_available():
  pytest.importorskip("jax")
  pytest.importorskip("tunix")
  from kwtfs import runtime  # noqa: F401
  from kwtfs.train import sft  # noqa: F401
