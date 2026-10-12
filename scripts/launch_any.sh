#!/usr/bin/env bash
# Zone-aware launch with spot-capacity retries, then keep the run's heartbeat alive.
#
#   scripts/launch_any.sh <run> <type> <rate_usd_h> <max_hours> <module> -- <module args...>
#
# Tries each zone in KWTFS_ZONES (default "us-east1-c us-west4-a") every 5 min,
# up to TRIES rounds (default 30). The first zone that passes the VM's
# self-delete gate wins; tpu.sh records it in runs/<run>/zone, and this script
# then becomes that run's heartbeat. Needs KWTFS_PROJECT and KWTFS_BUCKET (see tpu.sh).
#
# Example (one RL seed of the regret-reward run, E3b):
#   INPUT="data/kwtfs/rl.jsonl runs/sft/lora.npz" scripts/launch_any.sh rl-e3b-s0 v5litepod-4 2.0 1.5 kwtfs.train.grpo -- \
#     --config grpo_4b.yaml --init-lora /opt/kwtfs/in/lora.npz --mesh 4x1 \
#     --reward-off faithful+genre rollout_config.seed=0
set -uo pipefail
cd "$(dirname "$0")/.."
run=$1; type=$2; rate=$3; hours=$4; module=$5; shift 5; [ "${1:-}" = "--" ] && shift
zones=${KWTFS_ZONES:-"us-east1-c us-west4-a"}
for i in $(seq 1 "${TRIES:-30}"); do
  for z in $zones; do
    out=$(KWTFS_ZONE=$z TYPE=$type RATE=$rate MODULE=$module bash scripts/tpu.sh launch "$run" "$hours" -- "$@" 2>&1)
    if echo "$out" | grep -q "GUARD_OK: VM"; then
      echo "$run launched in $z (try $i) at $(date -u +%H:%M)"
      exec bash scripts/tpu.sh heartbeat "$run"
    fi
    echo "$run $z try $i: $(echo "$out" | grep -oE 'no more capacity|Reservation not found|Quota[^.]*|ERROR[^{]*' | head -1)"
  done
  sleep 300
done
echo "$run gave up"
