#!/usr/bin/env bash
#SBATCH --job-name=oe-tag-rank
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
git rev-parse HEAD
"$python_bin" -m unittest discover -s "$audit" -p test_tag_rank_metrics.py -v
"$python_bin" -u "$audit/oracle_closed_loop.py" --tag-ranking --run colored_square_episode_seed0 --queries 64 --budget 5 --candidates 32 --seed 3101 --out "results/tag-ranking-${SLURM_JOB_ID}"
"$python_bin" "$audit/tag_rank_metrics.py" "results/tag-ranking-${SLURM_JOB_ID}/report.json"
