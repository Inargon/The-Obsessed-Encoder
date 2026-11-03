#!/usr/bin/env bash
#SBATCH --job-name=oe-budget-reach
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled
export TMPDIR="${SLURM_TMPDIR:-/tmp}/oe-budget-${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$repo_root/leworldmodel"
"$python_bin" -m unittest discover -s additional_files/control_relevance_audit -p 'test_budget*.py' -v
git rev-parse HEAD
"$python_bin" -u additional_files/control_relevance_audit/train_budget_reachability.py \
  --dataset "$STABLEWM_HOME/datasets/pusht_expert_train.h5" \
  --output "results/budget-reachability-${SLURM_JOB_ID}" \
  --samples "${BUDGET_SAMPLES:-12000}" --epochs "${BUDGET_EPOCHS:-200}" --seed 73
