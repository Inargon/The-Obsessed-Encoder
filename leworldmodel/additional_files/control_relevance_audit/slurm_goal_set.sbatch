#!/usr/bin/env bash
#SBATCH --job-name=oe-goal-set
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
"$python_bin" -m unittest discover -s "$audit" -p test_oracle_closed_loop.py -v
"$python_bin" -m unittest discover -s "$audit" -p test_goal_agent_intervention.py -v
"$python_bin" -u "$audit/oracle_closed_loop.py" \
  --queries 2 --budget 25 --goal-agent-audit --goal-set-policy \
  --reference-report results/oracle-closed-loop-113759/report.json \
  --out "results/goal-set-${SLURM_JOB_ID:?Submit through sbatch}"
