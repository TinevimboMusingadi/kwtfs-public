# kwtfs: knowing when to think fast or slow

One small language model learns, question by question, whether to **decide in a single forward pass**, **answer in a
few words**, or **reason first**. The model is Qwen3-4B with a LoRA adapter, trained with Google's
[Tunix](https://github.com/google/tunix) (JAX) on spot TPUs. Each mode is a tag the model writes, and the runtime
executes it natively. A one-step decision reads the option-letter logits and generates nothing. A short answer gets
a 48-token budget. Reasoning gets 256 tokens and can hand off to a one-step decision once it is ready.

Routing labels come from measuring the base model in every mode (capability profiling), not from task names. A short
SFT teaches the model to obey forced modes and to route. GRPO with a **regret reward** then improves the routing.
The regret reward scores the chosen mode against the measured best mode, and its weight is annealed to zero. Across
5 seeds it matches always-reasoning accuracy with about 30% fewer tokens, and it is far more stable than an
outcome-only reward.
Write-up: **https://tinevimbomusingadi.github.io/kwtfs-blog/**

## The mode format

The system prompt carries a control: `Mode: auto|jev|direct|cot`. The completion is a genre tag followed by mode
segments:

```
<genre:compare><mode:jev>\nB                                  jev: one option letter, read from logits (no generation)
<genre:arith><mode:direct>\n<answer>42</answer>                direct: short answer, cap 48 tokens
<genre:arith><mode:cot>\n6*7=42\n<answer>42</answer>           cot: reasoning, cap 256 tokens ...
<genre:wordproblem><mode:cot>\n...\n<mode:jev>\nC              ... which may hand off to a one-step decision
```

Genres: `decision, compare, rubric, arith, wordproblem, knowledge`. JEV only exists when lettered options are shown
(A–F, each one Qwen token). Parsing is strict (`kwtfs/format.py`). Under a forced control, the first mode must
match the control.

Labels use the utility **U = p − λt**, where p is accuracy, t is generated tokens and λ = 0.0005, so 100 tokens cost
5 accuracy points. The suggested mode for a question is argmax U over modes.

## Results (sealed test split, n = 799; 5 seeds per RL arm)

| model | auto accuracy | tokens / question | same model forced to CoT |
|---|---|---|---|
| SFT only (1 run) | 78.1% | 40.1 | 87.7% at 88.8 tokens |
| GRPO, outcome-only reward (E2) | 85.7% ± 3.2 | 47.5 | 88.6% at 81.2 tokens |
| **GRPO, regret reward (E3b)** | **88.6% ± 0.3** | **54.4** | 88.5% at 78.5 tokens |

The ± values are the standard deviation over 5 seeds. E3b − E2 = +2.9 points (one-sided exact permutation test
over seeds, p = 0.008), and E3b's seed SD is 12× smaller. Under auto routing, E3b uses 31% fewer tokens than when it
is forced to reason. On the held-out OOD split (n = 460: an unseen decision domain plus 6-digit arithmetic), the
numbers are 78.3% ± 0.8 for E3b and 77.3% ± 0.9 for E2 (p = 0.07). Forced-mode obedience is 100% in every
evaluation. The post has the full set of figures, ablations and caveats. These values come from the post's
`data/numbers.json`.

**Cost.** The whole project ran on spot TPU v5e for roughly US$40: single chips (v5litepod-1) for profiling, SFT
and evaluation, and v5litepod-4 for RL (150 GRPO steps, about 1 h per seed).

## Repository layout

```
kwtfs/
  format.py        mode/genre tags, chat prompt, strict parser, exact grading
  rewards.py       GRPO reward terms (Tunix sums every public function); the regret reward
  tasks.py         Tunix data_module: RL prompts with 30% forced controls
  checks.py        verifies `expr = number` claims in reasoning traces
  mode_value.py    hidden-state probe predicting p(correct | question, mode) and tokens
  runtime.py       Tunix sampler + option-letter ids shared by eval and profiling
  data/            sources -> schema -> dedup -> build -> profile -> labels -> prep -> sft_data
  train/sft.py     SFT warm-start (tag-token upweighting, completion-window logits)
  train/grpo.py    stock Tunix GRPO (+ model loading, LoRA save/load)
  eval/evaluate.py native execution under every control; scoring; batch-1 latency
  eval/report.py   re-scoring, paired bootstrap CIs, token-price sweep
configs/grpo_4b.yaml   Tunix overrides (Qwen3-4B, LoRA r16, bf16, 150 steps)
scripts/tpu.sh         headless spot TPU launch / heartbeat / logs / fetch; every VM deletes itself
scripts/vm_startup.sh  VM side: self-delete gate, kill switches, pinned install, job
scripts/launch_any.sh  zone-aware launch with spot-capacity retries
notebooks/             CPU quickstart (+ optional TPU smoke run) and a small label sample
docs/data/             data source and licence register, gold-audit decisions
tests/                 CPU-only tests
```

## Quickstart (CPU)

```bash
pip install -e ".[dev]"
pytest
jupyter notebook notebooks/01_tunix_quickstart.ipynb
```

The notebook covers four things: the parser on example completions, the reward terms with the exploit tests
(disobeying a forced mode, false equations, short-and-wrong against long-and-right, annealed regret), a toy RLVR
problem showing why a correctness-only reward drifts to long reasoning, and λ relabelling of a bundled 360-item
profile sample. Its last section, disabled by default, runs a 3-step GRPO smoke on Qwen3-0.6B on a Colab or Kaggle
TPU.

## Full pipeline (TPU)

The exact environment used for every run is `requirements-tpu.lock` (Python 3.12, hash-pinned) plus Tunix at the
revision in `scripts/vm_startup.sh`. `pip install -e ".[tpu]"` gets close to it. Model weights download anonymously
from the Hugging Face Hub.

TPU jobs go through `scripts/tpu.sh`. It uploads the code, creates a spot VM, runs one module and uploads the
outputs. The VM then deletes itself when the job exits, when the log goes silent, at a wall-clock deadline, or when
the heartbeat from your machine stops. Set:

| variable | meaning |
|---|---|
| `KWTFS_PROJECT` | GCP project id (required) |
| `KWTFS_BUCKET` | GCS prefix for code, inputs, logs and outputs, e.g. `gs://my-bucket/kwtfs` (required) |
| `KWTFS_ZONE` | zone for `tpu.sh` (default `us-east1-c`); `KWTFS_ZONES` is the list `launch_any.sh` tries |
| `TYPE`, `RATE`, `MODULE`, `INPUT` | accelerator type, USD/h for the printed cost cap, module to run, files staged to `/opt/kwtfs/in/` |

The package itself reads these variables. The flags of `kwtfs.train.grpo` set them for you:

| variable | flag | default |
|---|---|---|
| `KWT_CONFIG` | `--config` | `grpo_4b.yaml` (a name in `configs/` or a path) |
| `KWT_REWARD_OFF` | `--reward-off` | none; e.g. `faithful+genre` (E3b), `route_regret+faithful+genre` (E2) |
| `KWT_BETA` | `--beta` | `0.5` |
| `KWT_UTILITIES` | `--utilities` | `utilities` (probe column); `label` = measured profile (E6) |
| `KWT_ANNEAL_COMPLETIONS` | (none) | `2880` (β reaches 0 after 60% of training) |
| `KWT_RL_DATA` | (none) | `/opt/kwtfs/in/rl.jsonl` |

Every TPU line below is `scripts/tpu.sh launch <run> <max_hours> -- <module args>`. Follow each one with
`scripts/tpu.sh heartbeat <run>`, and run `scripts/tpu.sh fetch <run>` when it finishes. The VM adds
`--out /opt/kwtfs/out` itself. Each `--items` default (`in/...`) resolves to the files staged with `INPUT`.

```bash
# 1. data (CPU): download the train splits listed in docs/data/SOURCES.md into data/kwtfs/raw/<name>/
#    (e.g. `huggingface-cli download openai/gsm8k --repo-type dataset --local-dir data/kwtfs/raw/gsm8k`;
#     dirs: bev-decision, gsm8k, aqua_rat, ai2_arc, openbookqa, commonsense_qa), then:
python -m kwtfs.data.build                        # -> data/kwtfs/items.jsonl, build_report.json
python -c "from kwtfs.data import build; build.drop_long(tokenizer_json='<Qwen3 tokenizer.json>')"

# 2. capability profiling (TPU, v5litepod-1): every item in every feasible mode, K=4, plus the gold audit
INPUT=data/kwtfs/items.jsonl MODULE=kwtfs.data.profile scripts/tpu.sh launch profile 7 -- --k 4

# 3. labels (CPU): report the label-quality gates, write the audit queue, apply reviewed fixes, write labels
python -m kwtfs.data.labels --profile runs/profile/profile.jsonl --report --queue runs/profile/audit_queue.jsonl
python -c "from kwtfs.data import build; build.apply_audit('docs/data/audit_decisions.json', profile_path='runs/profile/profile.jsonl')"
python -m kwtfs.data.labels --profile runs/profile/profile.jsonl --write data/kwtfs/labelled.jsonl

# 4. probe features + self-distilled CoT traces (TPU), then the mode-value probe and SFT data (CPU)
INPUT=data/kwtfs/items.jsonl MODULE=kwtfs.data.prep scripts/tpu.sh launch prep 4
python -m kwtfs.mode_value train --hidden runs/prep/hidden.npz --profile runs/profile/profile.jsonl --out runs/mode_value
python -m kwtfs.data.sft_data --profile runs/profile/profile.jsonl --traces runs/prep/traces.jsonl --out data/kwtfs/sft.jsonl

# 5. SFT (TPU, v5litepod-1)
INPUT=data/kwtfs/sft.jsonl MODULE=kwtfs.train.sft scripts/tpu.sh launch sft 2.5 -- \
  --examples /opt/kwtfs/in/sft.jsonl --model qwen3-4b=Qwen/Qwen3-4B

# 6. GRPO (TPU, v5litepod-4), one seed of the regret-reward arm; rl.jsonl = the rl split of labelled.jsonl
#    (the runs also dropped the ~3% of rows whose prompt exceeds rollout_config.max_prompt_length = 448 tokens)
INPUT="data/kwtfs/rl.jsonl runs/sft/lora.npz" scripts/launch_any.sh rl-e3b-s0 v5litepod-4 2.0 1.5 kwtfs.train.grpo -- \
  --config grpo_4b.yaml --init-lora /opt/kwtfs/in/lora.npz --mesh 4x1 --reward-off faithful+genre rollout_config.seed=0

# 7. evaluation (TPU, v5litepod-1): every control, several adapters on one loaded base, then report (CPU)
INPUT="data/kwtfs/labelled.jsonl sft.npz e3b.npz" MODULE=kwtfs.eval.evaluate scripts/tpu.sh launch final 2 -- \
  --loras sft=/opt/kwtfs/in/sft.npz e3b=/opt/kwtfs/in/e3b.npz --splits test ood
python -m kwtfs.eval.report --runs runs/final --split test --ref e3b
python -m kwtfs.eval.report --runs runs/final --split test --ref e3b --sweep 0 0.0005 0.001 0.002
```

**Regret utilities.** `route_regret` reads a `utilities` column (`{mode: U}`) on each RL row and falls back to the
row's measured label (`label.u`). E3b used utilities predicted by the mode-value probe. The exact values it trained
on are released as the `rl_utility_{jev,direct,cot}` columns of the `kwtfs-mode-labels` dataset on Hugging Face;
copy them into each RL row as `utilities` to reproduce E3b. (The one-off script that first wrote them is not part of
this package, and a fresh probe fit gives somewhat different values.) Without the column, step 6 trains on the
measured-profile utilities, which is the E6 arm (`--utilities label`). The post reports the two as performing the
same.

`python -m kwtfs.eval.evaluate ... --latency 160` times 160 test items at batch 1 instead of scoring them. For a
local plumbing check, `kwtfs.train.grpo`, `kwtfs.train.sft` and `kwtfs.data.profile` accept `--random-weights`
(a tiny random Qwen3). `kwtfs.train.grpo` also accepts `--smoke` (Qwen3-0.6B, 2 steps) and `--dry-run`.

## Data licences

Only the train splits of public benchmarks are used. The sealed test and OOD splits are carved out of that pool.
Raw downloads are never redistributed: `data/` is gitignored. Questions from the **BEV decision mix**
(`avbiswas/bev-decision`, mixed upstream sources, no blanket licence) and **OpenBookQA** (licence unknown) are
referenced only by item ID and statistics. The same applies to ARC (CC-BY-SA-4.0, but AI2 asks that it not be
redistributed outside AI2). GSM8K and CommonsenseQA are MIT, AQuA-RAT is Apache-2.0, and the procedural tasks are
generated by this code. The full register is in
`docs/data/SOURCES.md`. `notebooks/sample_labels.json` holds per-mode accuracy and token statistics only, with no
question text.

## Citation

```bibtex
@misc{musingadi2026kwtfs,
  author = {Musingadi, Tinevimbo},
  title  = {Knowing When to Think Fast or Slow at Test Time},
  year   = {2026},
  url    = {https://tinevimbomusingadi.github.io/kwtfs-blog/}
}
```

Code: Apache-2.0 (see `LICENSE`). Author: Tinevimbo Musingadi.
