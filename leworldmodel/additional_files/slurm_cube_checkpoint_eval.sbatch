#!/usr/bin/env bash
#SBATCH --job-name=oe-cube-eval
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?REPO_ROOT or SLURM_SUBMIT_DIR is required}}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export MUJOCO_GL=egl
unset PYOPENGL_PLATFORM || true
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_root/leworldmodel"
out_dir="$repo_root/leworldmodel/results/cube-checkpoint-eval-10k-seed0"
mkdir -p "$out_dir"

evaluate() {
  local run_name="$1"
  local tagged="$2"
  local extra=()
  if [[ "$tagged" == "true" ]]; then
    extra+=(--tagged)
  fi
  "$python_bin" additional_files/evaluate_cube_checkpoint.py \
    --run-name "$run_name" \
    --checkpoint weights_step_10000.pt \
    --num-eval 50 \
    --seed 0 \
    --output "$out_dir/${run_name}.json" \
    "${extra[@]}"
}

evaluate cube_clean_jepa_seed0 false
evaluate cube_clean_aligned_seed0 false
evaluate cube_tagged_jepa_seed0 true
evaluate cube_tagged_aligned_seed0 true

echo "[cube-eval] complete"
