#!/usr/bin/env bash
#SBATCH --job-name=oe-effect-set
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
# Keep the established cluster allocation recipe; this small NumPy pilot uses CPUs.
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONNOUSERSITE=1
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
"$python_bin" -m unittest discover -s "$audit" -p 'test_effect_set_pilot.py' -v
"$python_bin" -u "$audit/effect_set_pilot.py" \
  --input-dir "${FIXTURE_DIR:-$repo_root/leworldmodel/results/pair-goals-112328/fixture}" \
  --out "$repo_root/leworldmodel/results/effect-set-${SLURM_JOB_ID:?Submit through sbatch}/report.json"
