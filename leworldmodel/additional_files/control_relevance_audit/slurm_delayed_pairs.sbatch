#!/usr/bin/env bash
#SBATCH --job-name=oe-delayed-pairs
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
"$python_bin" -m unittest discover -s "$audit" -p 'test_*.py' -v
"$python_bin" -u "$audit/collect_delayed_pairs.py" --pairs 32 --out "$repo_root/leworldmodel/results/delayed-pairs-${SLURM_JOB_ID:?Submit through sbatch}"
