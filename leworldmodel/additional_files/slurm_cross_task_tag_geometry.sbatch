#!/usr/bin/env bash
#SBATCH --job-name=oe-tag-geom
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?REPO_ROOT or SLURM_SUBMIT_DIR is required}}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export PYTHONPATH="$repo_root:$repo_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_root/leworldmodel"
out_dir="$repo_root/leworldmodel/results/cross-task-tag-geometry"
mkdir -p "$out_dir"

"$python_bin" additional_files/diagnose_generic_tag_geometry.py \
  --dataset tworoom.h5 \
  --checkpoint clean_jepa=tworoom_clean_jepa_seed0/weights_step_10000.pt \
  --checkpoint clean_aligned=tworoom_clean_aligned_seed0/weights_step_10000.pt \
  --checkpoint tagged_jepa=tworoom_tagged_jepa_seed0/weights_step_10000.pt \
  --checkpoint tagged_aligned=tworoom_tagged_aligned_seed0/weights_step_10000.pt \
  --out "$out_dir/tworoom.json"

"$python_bin" additional_files/diagnose_generic_tag_geometry.py \
  --dataset ogbench/cube_single_expert.h5 \
  --checkpoint clean_jepa=cube_clean_jepa_full_seed0/weights_epoch_10.pt \
  --checkpoint clean_aligned=cube_clean_aligned_full_seed0/weights_epoch_10.pt \
  --checkpoint tagged_jepa=cube_tagged_jepa_full_seed0/weights_epoch_10.pt \
  --checkpoint tagged_aligned=cube_tagged_aligned_full_seed0/weights_epoch_10.pt \
  --out "$out_dir/cube.json"

echo "[cross-task-tag-geometry] complete"
