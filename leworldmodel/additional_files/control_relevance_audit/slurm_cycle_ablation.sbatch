#!/usr/bin/env bash
#SBATCH --job-name=oe-cycle-pair
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export TMPDIR="/grp01/ids_compcog/song/tmp/aluo/cycle-${SLURM_JOB_ID:?}"
export CYCLE_CAMPAIGN="job${SLURM_JOB_ID}"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
mkdir -p "$SPT_CACHE_DIR" "$TMPDIR"
cd "$repo_root/leworldmodel"
steps="${CYCLE_STEPS:-20}"
out="results/cycle-ablation-${SLURM_JOB_ID}"
"$python_bin" -m unittest discover -s additional_files/control_relevance_audit -p test_cycle_ablation.py -v
"$python_bin" -u additional_files/run_cycle_ablation.py --seeds 1 --gpus 0 --epochs 10 \
  --configs "cycle_on_${CYCLE_CAMPAIGN},cycle_off_${CYCLE_CAMPAIGN}" --results-dir "$out" \
  --extra-opts "++trainer.max_steps=$steps ++trainer.log_every_n_steps=1 ++eval.every_n_steps=100000000 ++checkpoint.every_n_steps=$steps"
printf 'CYCLE_PAIR_COMPLETE steps_per_arm=%s results=%s\n' "$steps" "$out"
