#!/bin/bash
#SBATCH --job-name=oe-probe
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-oe-probe-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-aligned-ablations}"
python_bin="${PYTHON_BIN:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
dataset_dir="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"

cd "$repo_root"
export PYTHONPATH="$repo_root:$repo_root/leworldmodel"
export LOCAL_DATASET_DIR="$dataset_dir"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"

"$python_bin" leworldmodel/additional_files/diagnose_frozen_representation.py \
  --checkpoint aligned_peak_near96k=control_aligned_pred1_seed0/weights_epoch_7.pt \
  --checkpoint aligned_final=control_aligned_pred1_seed0/weights_epoch_10.pt \
  --checkpoint pred03_peak_near54k=masked_sequence_pred03_seed0/weights_epoch_4.pt \
  --checkpoint pred03_final=masked_sequence_pred03_seed0/weights_epoch_10.pt \
  --checkpoint masked_reachability_final=masked_reachability_seed0/weights_epoch_10.pt \
  --num-clips 512 \
  --out leworldmodel/results/frozen-representation/probes.json
