#!/usr/bin/env bash
# Spot TPU runs, headless: no SSH or scp needed. Every VM deletes itself.
#
#   scripts/tpu.sh launch <run> [max_hours] [-- module args]   upload code, create VM, start job
#   scripts/tpu.sh heartbeat <run>     keep running while the job runs; if this machine stops, the VM dies
#   scripts/tpu.sh logs <run> [lines]  tail the job log (synced every ~5 min)
#   scripts/tpu.sh status              list TPU VMs in the zone (want: empty)
#   scripts/tpu.sh delete <run>        delete now
#   scripts/tpu.sh fetch <run>         download outputs to runs/<run>
#
# Required env:
#   KWTFS_PROJECT   GCP project id
#   KWTFS_BUCKET    GCS prefix for code, inputs, logs and outputs, e.g. gs://my-bucket/kwtfs
# Optional env:
#   KWTFS_ZONE      zone (default us-east1-c). `launch` records the zone it used in
#                   runs/<run>/zone, and the other commands read it back, so a run
#                   launched by scripts/launch_any.sh is always found in the right zone.
#   TYPE            accelerator type (default v5litepod-1; RL used v5litepod-4)
#   VERSION         TPU software version (default v2-alpha-tpuv5-lite)
#   RATE            USD/hour for the whole slice, for the printed cost cap (default 0.50)
#   MODULE          python module to run (default kwtfs.train.grpo)
#   INPUT           space-separated local files / gs:// URIs staged to the VM's /opt/kwtfs/in/
#   NO_HEARTBEAT=1  disable the heartbeat kill switch
set -euo pipefail
cd "$(dirname "$0")/.."

: "${KWTFS_PROJECT:?set KWTFS_PROJECT to your GCP project id}"
: "${KWTFS_BUCKET:?set KWTFS_BUCKET, e.g. gs://my-bucket/kwtfs}"
PROJECT=$KWTFS_PROJECT
BUCKET=${KWTFS_BUCKET%/}
TYPE=${TYPE:-v5litepod-1}
VERSION=${VERSION:-v2-alpha-tpuv5-lite}
RATE=${RATE:-0.50}  # spot v5e was about $0.49 per chip-hour when this was written
SPEND=runs/spend.csv

cmd=${1:-}; shift || true
run=${1:-}; [ -n "$run" ] && shift || true
gcs="$BUCKET/$run"
vm="kwtfs-$run"
zone_file="runs/$run/zone"
if [ -n "$run" ] && [ "$cmd" != launch ] && [ -f "$zone_file" ]; then
  ZONE=$(cat "$zone_file")
else
  ZONE=${KWTFS_ZONE:-us-east1-c}
fi

case "$cmd" in
launch)
  hours=${1:-3}; shift || true; [ "${1:-}" = "--" ] && shift
  args="$*"
  deadline=$(awk "BEGIN{printf \"%d\", $(date +%s) + $hours*3600}")
  echo "max cost: $(awk "BEGIN{print $hours*$RATE}") USD ($TYPE in $ZONE, ${hours}h at \$$RATE/h)"
  tmp=$(mktemp -d)
  tar --exclude=__pycache__ -czf "$tmp/code.tar.gz" kwtfs configs requirements-tpu.lock
  gcloud storage cp "$tmp/code.tar.gz" "$gcs/code.tar.gz"
  # INPUT: space-separated local files or gs:// URIs, staged to <run>/in/ (the VM sees /opt/kwtfs/in/<name>).
  for f in ${INPUT:-}; do gcloud storage cp "$f" "$gcs/in/"; done
  heartbeat=""
  if [ -z "${NO_HEARTBEAT:-}" ]; then
    heartbeat="$gcs/heartbeat"; date +%s | gcloud storage cp - "$heartbeat"
  fi
  # A failed create (e.g. no spot capacity) exits non-zero here: nothing was started.
  gcloud compute tpus tpu-vm create "$vm" --project="$PROJECT" --zone="$ZONE" \
    --accelerator-type="$TYPE" --version="$VERSION" --spot --labels=kwtfs=1 \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="kwtfs-name=$vm,kwtfs-zone=$ZONE,kwtfs-project=$PROJECT,kwtfs-deadline=$deadline,kwtfs-gcs=$gcs,kwtfs-heartbeat=$heartbeat,kwtfs-args=$args,kwtfs-module=${MODULE:-kwtfs.train.grpo}" \
    --metadata-from-file=startup-script=scripts/vm_startup.sh
  mkdir -p "runs/$run"
  echo "$ZONE" > "$zone_file"
  [ -f "$SPEND" ] || echo "run,type,zone,started_utc,max_hours,rate_usd_h,max_usd" > "$SPEND"
  echo "$run,$TYPE,$ZONE,$(date -u +%FT%TZ),$hours,$RATE,$(awk "BEGIN{print $hours*$RATE}")" >> "$SPEND"
  echo "waiting for the self-delete gate (GUARD_OK)..."
  for _ in $(seq 1 40); do
    if gcloud storage cat "$gcs/logs/GUARD_OK" >/dev/null 2>&1; then
      echo "GUARD_OK: VM can delete itself. Now run: scripts/tpu.sh heartbeat $run"; exit 0
    fi
    sleep 15
  done
  echo "no GUARD_OK after 10 min; deleting $vm to be safe"
  "$0" delete "$run"; exit 1
  ;;
heartbeat)
  [ -n "${NO_HEARTBEAT:-}" ] && { echo "heartbeat disabled"; exit 0; }
  echo "heartbeat for $vm ($ZONE) every 5 min (Ctrl-C stops it; the VM then deletes itself within ~30 min)"
  # Stop only when the VM is confirmed gone (NOT_FOUND). Ending on a network
  # blip would let a healthy VM delete itself 30 min later.
  while true; do
    if ! out=$(gcloud compute tpus tpu-vm describe "$vm" --zone="$ZONE" --project="$PROJECT" \
          --format='value(state)' 2>&1); then
      echo "$out" | grep -q "NOT_FOUND" && break
    fi
    date +%s | gcloud storage cp - "$gcs/heartbeat" >/dev/null 2>&1 || true
    sleep 300
  done
  echo "$vm is gone; heartbeat stopped"
  ;;
logs)
  gcloud storage cat "$gcs/logs/REASON" 2>/dev/null && echo "(VM finished, reason above)"
  gcloud storage cat "$gcs/logs/job.log" 2>/dev/null | tail -n "${1:-60}"
  ;;
status)
  gcloud compute tpus tpu-vm list --zone="$ZONE" --project="$PROJECT" \
    --format='table(name,acceleratorType,state,createTime,labels.kwtfs)'
  ;;
delete)
  gcloud compute tpus tpu-vm delete "$vm" --zone="$ZONE" --project="$PROJECT" --quiet || true
  KWTFS_ZONE=$ZONE "$0" status
  ;;
fetch)
  mkdir -p "runs/$run"
  gcloud storage cp -r "$gcs/out/*" "$gcs/logs" "runs/$run/"
  ;;
*)
  sed -n 2,24p "$0"; exit 1
  ;;
esac
