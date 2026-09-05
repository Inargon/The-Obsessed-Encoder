#!/usr/bin/env bash
#SBATCH --job-name=oe-mask-pred
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=144G
#SBATCH --time=06:00:00
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

# Three seed-0 arms, not three seeds. The 2000-step boundary sweep showed that
# weight 1.0 relearns the tag shortcut and weight 0.0 loses world prediction.
# This longer interior sweep searches for the trade-off's useful operating
# point; the six-hour request includes headroom for shuffle diagnostics and
# repeated CEM evaluations.
"$python_bin" additional_files/run_control.py \
  --configs masked_sequence_pred03,masked_sequence_pred05,masked_sequence_pred07 \
  --seeds 1 \
  --gpus 0,1,2 \
  --results-dir "$repo_root/leworldmodel/results/masked-pred-mid-sweep" \
  --extra-opts '+trainer.max_steps=10000 +trainer.log_every_n_steps=10 ++eval.every_n_steps=2000 ++pair.every_n_steps=1000 ++checkpoint.every_n_steps=5000'
