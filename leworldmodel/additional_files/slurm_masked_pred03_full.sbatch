#!/usr/bin/env bash
#SBATCH --job-name=oe-mask03-s0
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export TMPDIR="${TMPDIR:-/grp01/ids_compcog/song/tmp}"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SPT_CACHE_DIR" "$TMPDIR"
cd "$repo_root/leworldmodel"

# Seed-0 promotion of the only arm whose 10k-step success curve was monotonic.
# This starts clean because the LeWM runner does not reliably resume optimizer,
# scheduler, and logging state from the mid-sweep checkpoint.
"$python_bin" additional_files/run_control.py \
  --configs masked_sequence_pred03 \
  --seeds 1 \
  --gpus 0 \
  --results-dir "$repo_root/leworldmodel/results/masked-pred03-full-seed0" \
  --extra-opts '++checkpoint.every_n_steps=5000'
