#!/usr/bin/env bash
#SBATCH --job-name=oe-geom-smoke
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-effect-geometry}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export TMPDIR="${TMPDIR:-/grp01/ids_compcog/song/tmp}"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SPT_CACHE_DIR" "$TMPDIR"
cd "$repo_root/leworldmodel"

bank=results/effect-geometry-bank/bank.npz
if [[ ! -f "$bank" ]]; then
  "$python_bin" additional_files/generate_effect_geometry.py \
    --out "$bank" --anchors 128 --candidates 48 --branches 16
fi
"$python_bin" additional_files/run_control.py \
  --configs effect_geometry_pred03 --seeds 1 --gpus 0 \
  --results-dir "$repo_root/leworldmodel/results/effect-geometry-smoke" \
  --extra-opts '+trainer.max_steps=20 ++eval.every_n_steps=100000 ++pair.every_n_steps=100000'
