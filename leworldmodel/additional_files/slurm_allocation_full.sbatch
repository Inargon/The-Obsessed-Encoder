#!/usr/bin/env bash
#SBATCH --job-name=oe-alloc-full
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=144G
#SBATCH --time=2-00:00:00
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

# Nine runs total (three methods x three seeds), dispatched one at a time to
# each of the three allocated GPUs.
"$python_bin" additional_files/run_allocations.py \
  --seeds 3 \
  --gpus 0,1,2 \
  --results-dir "$repo_root/leworldmodel/results/allocation-full"
