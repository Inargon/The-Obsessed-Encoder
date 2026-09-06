#!/usr/bin/env bash
#SBATCH --job-name=oe-align-confirm
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=144G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-aligned-grad}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export TMPDIR="${TMPDIR:-/grp01/ids_compcog/song/tmp}"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SPT_CACHE_DIR" "$TMPDIR"
cd "$repo_root/leworldmodel"

# Two confirmation seeds and the exact ordinary-gradient comparator run in
# parallel. Run names and result roots are disjoint from the completed seed 0.
"$python_bin" additional_files/run_control.py \
  --configs control_aligned_pred1 \
  --seeds 1,2 \
  --gpus 0,1 \
  --results-dir "$repo_root/leworldmodel/results/aligned-grad-confirm-seeds12" \
  --extra-opts '++checkpoint.every_n_steps=5000' &
aligned_pid=$!

"$python_bin" additional_files/run_control.py \
  --configs masked_sequence_pred1 \
  --seeds 0, \
  --gpus 2 \
  --results-dir "$repo_root/leworldmodel/results/masked-sequence-pred1-full-seed0" \
  --extra-opts '++checkpoint.every_n_steps=5000' &
baseline_pid=$!

set +e
wait "$aligned_pid"
aligned_status=$?
wait "$baseline_pid"
baseline_status=$?
set -e

if (( aligned_status != 0 || baseline_status != 0 )); then
  echo "confirmation failed: aligned=$aligned_status baseline=$baseline_status" >&2
  exit 1
fi
