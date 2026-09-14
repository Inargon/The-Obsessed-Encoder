#!/usr/bin/env bash
#SBATCH --job-name=oe-reach-eval
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?REPO_ROOT or SLURM_SUBMIT_DIR is required}}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
checkpoint="${CHECKPOINT:-weights_step_5000.pt}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export MUJOCO_GL=egl
unset PYOPENGL_PLATFORM || true
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_root/leworldmodel"
step="${checkpoint#weights_step_}"
step="${step%.pt}"
out_dir="$repo_root/leworldmodel/results/reacher-checkpoint-eval-${step}-seed0"
mkdir -p "$out_dir"

evaluate_if_present() {
  local run_name="$1"
  local tagged="$2"
  local checkpoint_path="$STABLEWM_HOME/checkpoints/$run_name/$checkpoint"
  if [[ ! -f "$checkpoint_path" ]]; then
    echo "[reacher-eval] skip missing $checkpoint_path"
    return 0
  fi
  local extra=()
  if [[ "$tagged" == "true" ]]; then
    extra+=(--tagged)
  fi
  "$python_bin" additional_files/evaluate_reacher_checkpoint.py \
    --run-name "$run_name" \
    --checkpoint "$checkpoint" \
    --num-eval 50 \
    --seed 0 \
    --output "$out_dir/${run_name}.json" \
    "${extra[@]}"
}

evaluate_if_present reacher_clean_aligned_seed0 false
evaluate_if_present reacher_tagged_aligned_seed0 true

echo "[reacher-eval] complete"
