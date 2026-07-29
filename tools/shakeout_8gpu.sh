#!/usr/bin/env bash
# One-hour shakeout of all three examples on an 8-GPU box.
#
# Purpose: exercise every run path end to end (training, wandb, metrics
# mirror, dense tail, figures) at a budget — NOT to reach the full operating
# points. GPU split: LeJEPA on 0-2 (one arm each), DINOv3 on 3-5 (one arm
# each), LeWM on 6-7 (four arms, two waves).
#
# First invocation on a fresh node also prepares the datasets (~20 min,
# all downloads in parallel); re-runs skip both the prep and any run whose
# summary.json exists. wandb: online by default under one group tag
# (WANDB_MODE=offline|disabled to silence).
#
# The split is hardcoded to 8 GPUs; there is no reduced-GPU mode.
#
#   bash tools/shakeout_8gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Override DATA_DIR/RESULTS_DIR to place data and runs on a large volume.
export DATA_DIR="${DATA_DIR:-./data}"
export RESULTS_ROOT="${RESULTS_ROOT:-${RESULTS_DIR:-./results}/shakeout}"
TAG="${TAG:-shakeout_$(date +%y%m%d_%H%M)}"
mkdir -p "$DATA_DIR" "$RESULTS_ROOT" logs
# The LeWM lane launches from leworldmodel/, so every inherited path must be
# absolute -- a relative DATA_DIR silently resolves against the child's cwd.
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
RESULTS_ROOT="$(cd "$RESULTS_ROOT" && pwd)"
export STABLEWM_HOME="$DATA_DIR/stable_worldmodel"
export HF_HOME="$DATA_DIR/hf_cache"

echo "== shakeout '$TAG' -> $RESULTS_ROOT (data under $DATA_DIR)"
uv sync --frozen >/dev/null

# ---------------------------------------------------------------- data prep
prep_in1k() {
  uv run python lejepa/additional_files/prepare_data.py \
      --dataset imagenet1k --data-dir "$DATA_DIR"
}
prep_nyu() {
  uv run python dinov3/additional_files/fetch_nyu.py
}
prep_lewm() {
  mkdir -p "$STABLEWM_HOME/datasets"
  local h5="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
  if [ ! -f "$h5" ]; then
    uv run hf download quentinll/lewm-pusht pusht_expert_train.h5.zst \
        --repo-type dataset --local-dir "$STABLEWM_HOME/expert_archive"
    uv run python - <<PY
import zstandard, shutil
with open("$STABLEWM_HOME/expert_archive/pusht_expert_train.h5.zst", "rb") as src, \
     open("$h5.tmp", "wb") as dst:
    zstandard.ZstdDecompressor().copy_stream(src, dst)
shutil.move("$h5.tmp", "$h5")
print("expert set ready")
PY
  fi
  # Collect into a scratch name and rename on success: an interrupted collect
  # must not leave a half-written table that the existence check then trusts.
  if [ ! -d "$STABLEWM_HOME/datasets/pusht_scripted_goal_train.lance" ]; then
    # FULL production size: this same dataset feeds the full-length randgoal
    # runs afterwards, and a budget-sized set here silently trains them on a
    # fraction of the data (10 epochs over 2000 episodes finish in ~25 min and
    # look plausible). Budget-capped shakeout runs only read the head of the
    # stream, so full size costs the shakeout nothing but this one-time
    # collection (~30-60 min).
    rm -rf "$STABLEWM_HOME/datasets/pusht_scripted_goal_train_wip.lance"
    uv run python leworldmodel/additional_files/pusht_random_dest/collect.py \
        --name pusht_scripted_goal_train_wip --episodes 18685 --num-workers 64
    mv "$STABLEWM_HOME/datasets/pusht_scripted_goal_train_wip.lance" \
       "$STABLEWM_HOME/datasets/pusht_scripted_goal_train.lance"
  fi
}

echo "== preparing datasets (parallel; skipped where present)"
prep_in1k  > logs/prep_in1k.log  2>&1 &  P1=$!
prep_nyu   > logs/prep_nyu.log   2>&1 &  P2=$!
prep_lewm  > logs/prep_lewm.log  2>&1 &  P3=$!
FAILED=0
for p in $P1 $P2 $P3; do wait "$p" || FAILED=1; done
if [ "$FAILED" -ne 0 ]; then
  echo "!! data preparation failed -- see logs/prep_*.log"; exit 1
fi
echo "== datasets ready"

# ------------------------------------------------------------------ training
# Budgets sized for ~45 min of training on H100s; every run exercises its
# eval ticks, pair metrics, and (DINOv3) the dense tail at a small head cap.
# SHAKEOUT_QUICK=1 shrinks the budgets to ~15 min while keeping at least one
# eval tick / checkpoint / teacher milestone inside every run -- same path
# coverage, less soak (long-tail behaviour like late memory peaks needs the
# full budgets).
if [ "${SHAKEOUT_QUICK:-0}" = "1" ]; then
  LEJEPA_STEPS=2000     # one eval tick at 2000 exactly
  DINOV3_ITERS=1200     # milestone 999 + two probe val passes
  LEWM_STEPS=700
  LEWM_EXTRA="+trainer.max_steps=$LEWM_STEPS ++eval.every_n_steps=500"
  echo "== QUICK mode: budgets 2000/1200/700"
else
  LEJEPA_STEPS=6000
  DINOV3_ITERS=4000
  LEWM_EXTRA="+trainer.max_steps=2000"
fi

uv run python lejepa/additional_files/run.py \
    --seeds 1 --gpus 0,1,2 --max-steps "$LEJEPA_STEPS" \
    --results-dir "$RESULTS_ROOT/lejepa" --data-dir "$DATA_DIR" \
    --tag "$TAG" > logs/lejepa.log 2>&1 &  T1=$!

uv run python dinov3/additional_files/run.py \
    --seeds 1 --gpus 3,4,5 --max-iter "$DINOV3_ITERS" --head-iters 300 \
    --extra-opts "evaluation.eval_period_iterations=1000" \
    --results-dir "$RESULTS_ROOT/dinov3" --data-dir "$DATA_DIR" \
    --tag "$TAG" > logs/dinov3.log 2>&1 &  T2=$!

# uv resolves the project env in both layouts (repo .venv on bare metal,
# UV_PROJECT_ENVIRONMENT=/opt/venv in the shipped image -- which has no .venv).
(cd leworldmodel && PYTHONPATH=.. uv run --project .. python additional_files/run.py \
    --seeds 1 --gpus 6,7 --extra-opts "$LEWM_EXTRA" \
    --results-dir "$RESULTS_ROOT/lewm" --data-dir "$DATA_DIR" \
    --tag "$TAG") > logs/lewm.log 2>&1 &  T3=$!

echo "== training launched (logs/{lejepa,dinov3,lewm}.log; wandb group '$TAG')"
CODE=0
for name in lejepa:$T1 dinov3:$T2 lewm:$T3; do
  case_name="${name%%:*}"; pid="${name##*:}"
  if wait "$pid"; then
    echo "== $case_name: OK"
  else
    echo "!! $case_name: FAILED -- see logs/$case_name.log"; CODE=1
  fi
done

echo
echo "== figures:"
find "$RESULTS_ROOT" -path "*/figures/*" -name "*.png" | sort
echo "== wandb group: $TAG (per-example projects: obsessed-encoder-{lejepa,dinov3,lewm})"
exit $CODE
