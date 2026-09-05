#!/usr/bin/env bash
#SBATCH --job-name=oe-route-s0
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-action-router}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export TMPDIR="${TMPDIR:-/grp01/ids_compcog/song/tmp}"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SPT_CACHE_DIR" "$TMPDIR"
cd "$repo_root/leworldmodel"

# Full 10-epoch, seed-0 architectural bet. It retains the successful pred03
# objective and changes only how observed patch tokens reach the predictor.
"$python_bin" additional_files/run_control.py \
  --configs action_patch_pred03 \
  --seeds 1 \
  --gpus 0 \
  --results-dir "$repo_root/leworldmodel/results/action-router-full-seed0" \
  --extra-opts '++checkpoint.every_n_steps=5000'
