#!/usr/bin/env bash
#SBATCH --job-name=oe-visual-terminal
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
"$python_bin" -m unittest discover -s "$audit" -p 'test_*.py' -v
"$python_bin" -u "$audit/state_terminal_readout.py" --input-dir "$repo_root/leworldmodel/results/pair-goals-112328/fixture" --features-dir "$repo_root/leworldmodel/results/pair-goals-112328/readout" --out "$repo_root/leworldmodel/results/visual-terminal-${SLURM_JOB_ID:?Submit through sbatch}/report.json"
