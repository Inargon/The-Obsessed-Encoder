#!/usr/bin/env bash
#SBATCH --job-name=oe-comp-grad
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?REPO_ROOT or SLURM_SUBMIT_DIR is required}}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export TMPDIR="${TMPDIR:-/grp01/ids_compcog/song/tmp}"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SPT_CACHE_DIR" "$TMPDIR"
cd "$repo_root/leworldmodel"

"$python_bin" additional_files/run_component_gradient_diagnostic.py \
  --seeds 1 \
  --gpus 0 \
  --results-dir "$repo_root/leworldmodel/results/component-gradient-diagnostic-seed0" \
  --extra-opts 'trainer.max_epochs=1 +trainer.max_steps=6000 ++eval.every_n_steps=1000000 ++pair.every_n_steps=1000000 ++checkpoint.every_n_steps=25000'
