#!/usr/bin/env bash
#SBATCH --job-name=oe-bind-smoke
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-action-binding}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-$STABLEWM_HOME/datasets}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export TMPDIR="${TMPDIR:-/grp01/ids_compcog/song/tmp}"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SPT_CACHE_DIR" "$TMPDIR"
cd "$repo_root/leworldmodel"

bank=results/action-binding-bank/bank.npz
if [[ ! -s "$bank" ]]; then
  "$python_bin" additional_files/generate_action_binding.py \
    --out "$bank" --anchors 96 --candidates 64 --branches 8 --hard-pairs 1
fi
"$python_bin" additional_files/run_action_binding.py \
  --seeds 1 --gpus 0,1,2 \
  --results-dir "$repo_root/leworldmodel/results/action-binding-smoke" \
  --extra-opts '+trainer.max_steps=20 +trainer.log_every_n_steps=10 ++eval.every_n_steps=100000 ++pair.every_n_steps=100000'
