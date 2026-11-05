#!/usr/bin/env bash
#SBATCH --job-name=oe-jepa-original
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/oe-jepa-original-${USER}-${SLURM_JOB_ID:?Submit through sbatch}"
mkdir -p "$TMPDIR"
cd "$repo_root/leworldmodel"
out="results/jepa-original-${SLURM_JOB_ID}"
mkdir "$out"
git rev-parse HEAD > "$out/git-head.txt"
git status --short > "$out/git-status.txt"
cp config/eval/pusht.yaml "$out/pusht.yaml"
cp config/eval/solver/cem.yaml "$out/cem.yaml"
# Use the existing evaluator and its complete CEM configuration unchanged.
# Seed 42 is a declared fresh baseline check, not an asserted historical seed.
num_eval="${JEPA_NUM_EVAL:-50}"
[[ "$num_eval" =~ ^[1-9][0-9]*$ ]] || { echo 'JEPA_NUM_EVAL must be positive'; exit 1; }
"$python_bin" -u additional_files/evaluate_pusht_checkpoint.py \
  --run-name pusht_clean_jepa_matched_full_seed0 \
  --checkpoint weights_epoch_10.pt --seed 42 --num-eval "$num_eval" \
  --output "$out/report.json"
echo "ORIGINAL_JEPA_EVAL_COMPLETE $out/report.json"
