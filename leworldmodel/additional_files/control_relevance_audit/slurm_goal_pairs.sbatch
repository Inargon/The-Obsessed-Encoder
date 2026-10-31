#!/usr/bin/env bash
#SBATCH --job-name=oe-pair-goals
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
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
root="$repo_root/leworldmodel/results/pair-goals-${SLURM_JOB_ID:?Submit through sbatch}"
mkdir "$root"
"$python_bin" -m unittest discover -s "$audit" -p 'test_*.py' -v
"$python_bin" -u "$audit/collect_delayed_pairs.py" --pairs 256 --seed 902 --vary-goals --out "$root/fixture"
"$python_bin" -u "$audit/readout_delayed_pairs.py" --input-dir "$root/fixture" --out "$root/readout"
echo GOAL_PAIR_AUDIT_COMPLETE
