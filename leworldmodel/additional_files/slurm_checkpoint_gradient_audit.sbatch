#!/bin/bash
#SBATCH --job-name=oe-grad-audit
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-oe-grad-audit-%j.out

set -euo pipefail

repo="${REPO_ROOT:-$PWD}"
python_bin=/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python
checkpoint_root=/grp01/ids_compcog/song/swm/checkpoints

export STABLEWM_HOME=/grp01/ids_compcog/song/swm
export LOCAL_DATASET_DIR=/grp01/ids_compcog/song/swm
export PYTHONPATH="$repo:$repo/leworldmodel"
export MUJOCO_GL=egl
unset PYOPENGL_PLATFORM || true

cd "$repo/leworldmodel"
out="$repo/leworldmodel/results/fixed-batch-gradient-audit"
mkdir -p "$out"

run_audit() {
  local group="$1"
  shift
  local args=()
  local spec label relative
  for spec in "$@"; do
    label="${spec%%=*}"
    relative="${spec#*=}"
    if [[ -f "$checkpoint_root/$relative" ]]; then
      args+=(--checkpoint "$label=$relative")
    else
      echo "[skip] missing $checkpoint_root/$relative"
    fi
  done
  if (( ${#args[@]} < 4 )); then
    echo "[skip] $group has fewer than two checkpoints"
    return
  fi
  "$python_bin" additional_files/diagnose_checkpoint_gradient_geometry.py \
    "${args[@]}" \
    --num-batches 16 \
    --batch-size 32 \
    --seed 20260916 \
    --output "$out/$group.json"
}

run_audit pusht_tagged_10k \
  cosine=control_aligned_pred1_seed0/weights_step_10000.pt \
  conflict_only=pusht_tagged_conflict_only_pred1_seed0/weights_step_10000.pt \
  soft75=pusht_tagged_soft75_pred1_seed0/weights_step_10000.pt \
  orthogonal_raw=pusht_tagged_orthogonal_only_raw_pred1_seed0/weights_step_10000.pt \
  orthogonal_gated=pusht_tagged_orthogonal_only_gated_pred1_seed0/weights_step_10000.pt

run_audit reacher_clean_epoch1 \
  cosine=reacher_clean_aligned_diag_seed0/weights_epoch_1.pt \
  soft50=reacher_clean_soft50_diag_seed0/weights_epoch_1.pt \
  soft75=reacher_clean_soft75_diag_seed0/weights_epoch_1.pt \
  orthogonal_raw=reacher_clean_orthogonal_only_raw_seed0/weights_epoch_1.pt \
  orthogonal_gated=reacher_clean_orthogonal_only_gated_seed0/weights_epoch_1.pt

echo GRADIENT_AUDIT_COMPLETE
