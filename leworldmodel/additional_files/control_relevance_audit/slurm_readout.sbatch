#!/usr/bin/env bash
#SBATCH --job-name=oe-audit-readout
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
input="${AUDIT_INPUT_DIR:-$repo_root/leworldmodel/results/control-relevance-smoke-111924}"
out="$repo_root/leworldmodel/results/control-relevance-readout-${SLURM_JOB_ID:?Submit through sbatch}"
mkdir "$out"
"$python_bin" -m unittest discover -s "$audit" -p 'test_*.py' -v
"$python_bin" -u "$audit/multisplit_readout.py" --input-dir "$input" --out "$out/report.json" --seeds 73 74 75
