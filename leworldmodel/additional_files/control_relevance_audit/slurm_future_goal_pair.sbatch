#!/usr/bin/env bash
#SBATCH --job-name=oe-future-goal
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"
export FUTURE_GOAL_CAMPAIGN="job${SLURM_JOB_ID:?Submit through sbatch}"
# Node-local multiprocessing temporary files avoid the earlier shared-filesystem cleanup errors.
export TMPDIR="${SLURM_TMPDIR:-/tmp}/oe-future-goal-${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$SPT_CACHE_DIR"
cd "$repo_root/leworldmodel"
"$python_bin" -c 'import torch; print("TORCH", torch.__version__)'
"$python_bin" -m unittest discover -s additional_files/control_relevance_audit -p test_future_goal.py -v
steps="${FUTURE_GOAL_STEPS:-20}"
"$python_bin" -u additional_files/run_future_goal_pair.py \
  --configs "future_goal_detached_${FUTURE_GOAL_CAMPAIGN},future_goal_joint_${FUTURE_GOAL_CAMPAIGN}" \
  --seeds 1 --gpus 0 --epochs 10 --results-dir "results/future-goal-${SLURM_JOB_ID}" \
  --extra-opts "++trainer.max_steps=$steps ++trainer.log_every_n_steps=1 ++eval.every_n_steps=100000000 ++checkpoint.every_n_steps=$steps"
echo "FUTURE_GOAL_PAIR_COMPLETE steps_per_arm=$steps"
