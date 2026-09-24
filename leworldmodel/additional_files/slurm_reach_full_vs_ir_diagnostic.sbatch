#!/bin/bash
#SBATCH --job-name=oe-reach-full-ir
#SBATCH --partition=gpu_shared
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-oe-reach-full-ir-%A_%a.out

set -euo pipefail

repo="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin=/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python
out="$repo/leworldmodel/results/reach-full-vs-ir-diagnostic-20260924"

export STABLEWM_HOME=/grp01/ids_compcog/song/swm
export LOCAL_DATASET_DIR=/grp01/ids_compcog/song/swm
export SPT_CACHE_DIR=/grp01/ids_compcog/song/cache/stable-pretraining
export XDG_CACHE_HOME=/grp01/ids_compcog/song/cache/xdg
export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/${SLURM_JOB_ID}"
export PYTHONPATH="$repo:$repo/leworldmodel:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
unset PYOPENGL_PLATFORM || true

mkdir -p "$out" "$TMPDIR"
cd "$repo/leworldmodel"

common=(
  --num-batches 16
  --batch-size 32
  --seed 20260924
  --model-mode eval
)

case "$SLURM_ARRAY_TASK_ID" in
  0)
    group=tagged_pusht
    checkpoints=(
      --checkpoint ours=mechanism-20260920-020356_full_full_seed0/weights_epoch_10.pt
      --checkpoint inverse_reach=lctrl-ablation-seed0-quota-fixed-20260922_full_inverse_reach_seed0/weights_epoch_10.pt
    )
    ;;
  1)
    group=clean_reacher
    checkpoints=(
      --checkpoint ours=reacher_clean_aligned_full_seed0/weights_epoch_10.pt
      --checkpoint inverse_reach=clean-inverse-reach-20260923-084931_reacher_clean_inverse_reach_full_seed0/weights_epoch_10.pt
    )
    ;;
  *)
    echo "unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

"$python_bin" additional_files/diagnose_checkpoint_gradient_geometry.py \
  "${checkpoints[@]}" \
  "${common[@]}" \
  --output "$out/$group.json"

echo "REACH_FULL_VS_IR_DIAGNOSTIC_COMPLETE group=$group"
