#!/usr/bin/env bash
#SBATCH --job-name=oe-goal-role
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/oe-goal-role-${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
"$python_bin" -m unittest discover -s "$audit" -p test_goal_role.py -v
"$python_bin" -m unittest discover -s "$audit" -p test_goal_agent_intervention.py -v
git rev-parse HEAD
out="results/goal-role-${SLURM_JOB_ID:?Submit through sbatch}"
"$python_bin" -u "$audit/collect_goal_role.py" --output "$out/data" --queries "${GOAL_ROLE_QUERIES:-60}"
"$python_bin" -u "$audit/train_goal_role.py" --features "$out/data/features.npz" \
  --output "$out/readout" --epochs "${GOAL_ROLE_EPOCHS:-200}"
