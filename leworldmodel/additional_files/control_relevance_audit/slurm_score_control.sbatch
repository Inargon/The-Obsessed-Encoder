#!/usr/bin/env bash
#SBATCH --job-name=oe-score-control
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
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
"$python_bin" -m unittest discover -s "$audit" -p test_score_control.py -v
git rev-parse HEAD
"$python_bin" -u "$audit/oracle_closed_loop.py" \
  --physical-policy --queries 20 --budget 50 --candidates 8 --seed 2701 \
  --exclude-report results/oracle-closed-loop-113759/report.json \
  --out "results/score-control-${SLURM_JOB_ID:?Submit through sbatch}"
"$python_bin" "$audit/attribute_candidate_errors.py" \
  --reports "results/score-control-${SLURM_JOB_ID}/report.json" \
  --out "results/score-control-${SLURM_JOB_ID}/attribution.json"
echo "SCORE_CONTROL_COMPLETE"
