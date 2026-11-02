#!/usr/bin/env bash
#SBATCH --job-name=oe-response-plan
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
# NumPy CPU pilot; retains the established cluster allocation recipe.
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
"$python_bin" -m unittest discover -s "$audit" -p 'test_response*.py' -v
"$python_bin" -u "$audit/response_planning.py" \
  --input-dir "${FIXTURE_DIR:-$repo_root/leworldmodel/results/pair-goals-112328/fixture}" \
  --out "$repo_root/leworldmodel/results/response-planning-${SLURM_JOB_ID:?Submit through sbatch}/report.json"
