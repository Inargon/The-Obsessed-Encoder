#!/usr/bin/env bash
#SBATCH --job-name=oe-rg-probe
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"

checkpoint_root="$STABLEWM_HOME/checkpoints"

latest_checkpoint() {
  local run_name="$1"
  local checkpoint
  checkpoint=$(find "$checkpoint_root/$run_name" -maxdepth 1 -type f \
    \( -name 'weights_step_*.pt' -o -name 'weights_epoch_*.pt' \) \
    -printf '%T@ %f\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
  if [[ -z "$checkpoint" ]]; then
    echo "No checkpoint found for $run_name" >&2
    return 1
  fi
  printf '%s/%s' "$run_name" "$checkpoint"
}

old_run=randgoal_control_aligned_pred1_seed0
new_run=randgoal_decision_aligned_pred1_seed0
old_checkpoint=$(latest_checkpoint "$old_run")
new_checkpoint=$(latest_checkpoint "$new_run")

echo "old checkpoint: $old_checkpoint"
echo "new checkpoint: $new_checkpoint"

cd "$repo_root/leworldmodel"
"$python_bin" additional_files/diagnose_randgoal_start.py \
  --checkpoint "original_aligned=$old_checkpoint" \
  --checkpoint "decision_aligned=$new_checkpoint" \
  --dataset pusht_scripted_goal_train.lance \
  --cache-dir "$LOCAL_DATASET_DIR" \
  --num-episodes 2048 \
  --horizon 3 \
  --out results/randgoal-start-probe/probes.json
