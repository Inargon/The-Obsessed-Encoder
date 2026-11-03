#!/usr/bin/env bash
#SBATCH --job-name=oe-future-eval
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
export TMPDIR="${SLURM_TMPDIR:-/tmp}/oe-future-eval-${USER}-${SLURM_JOB_ID:?Submit through sbatch}"
mkdir -p "$TMPDIR"
cd "$repo_root/leworldmodel"
train_job="${FUTURE_GOAL_TRAIN_JOB:-114174}"
[[ "$train_job" =~ ^[0-9]+$ ]] || { echo 'Invalid training job ID'; exit 1; }
out="$repo_root/leworldmodel/results/future-goal-eval-${SLURM_JOB_ID}"
mkdir "$out"
"$python_bin" -m unittest discover -s additional_files/control_relevance_audit -p test_future_goal_eval.py -v
for arm in detached joint; do
  run="future_goal_${arm}_job${train_job}_seed0"
  test -f "$STABLEWM_HOME/checkpoints/$run/weights_step_10000.pt"
done
for arm in detached joint; do
  "$python_bin" -u additional_files/evaluate_pusht_checkpoint.py \
    --run-name "future_goal_${arm}_job${train_job}_seed0" \
    --checkpoint weights_step_10000.pt --seed 42 --num-eval 50 \
    --output "$out/$arm.json"
done
"$python_bin" additional_files/control_relevance_audit/summarize_future_goal_eval.py --directory "$out"
echo "FUTURE_GOAL_EVAL_COMPLETE $out"
