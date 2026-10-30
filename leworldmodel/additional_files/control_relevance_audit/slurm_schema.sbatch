#!/usr/bin/env bash
#SBATCH --job-name=oe-audit-schema
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
cd "$repo_root/leworldmodel"
"$python_bin" -u additional_files/control_relevance_audit/inspect_schema.py
