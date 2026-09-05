#!/usr/bin/env bash
#SBATCH --job-name=oe-mask-pred
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=144G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-/grp01/ids_compcog/song/cache/stable-pretraining}"
export TMPDIR="${TMPDIR:-/grp01/ids_compcog/song/tmp}"
export WANDB_MODE=disabled
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SPT_CACHE_DIR" "$TMPDIR"
cd "$repo_root/leworldmodel"

# Three seed-0 arms, not three seeds. Prior runs were still tied at 0.04 success
# on step 2000; 10000 steps reaches the first informative part of the learning
# curve while remaining roughly a one-hour pilot on three L40S GPUs.
"$python_bin" additional_files/run_control.py \
  --configs masked_sequence_pred1,masked_sequence_pred03,masked_sequence_pred0 \
  --seeds 1 \
  --gpus 0,1,2 \
  --results-dir "$repo_root/leworldmodel/results/masked-pred-sweep" \
  --extra-opts '+trainer.max_steps=10000 +trainer.log_every_n_steps=10 ++eval.every_n_steps=2000 ++pair.every_n_steps=1000 ++checkpoint.every_n_steps=5000'
