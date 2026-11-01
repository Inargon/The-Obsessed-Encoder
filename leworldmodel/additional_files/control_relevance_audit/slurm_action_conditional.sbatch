#!/usr/bin/env bash
#SBATCH --job-name=oe-action-cond
#SBATCH --partition=gpu_shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
"$python_bin" -m unittest discover -s "$audit" -p test_action_information.py -v
"$python_bin" -u "$audit/action_information.py" \
  --cache "${ACTION_CACHE:-results/action-information-112664}" \
  --out "results/action-conditional-${SLURM_JOB_ID:?Submit through sbatch}"
