#!/usr/bin/env bash
#SBATCH --job-name=oe-geom-diag
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
audit_root="${AUDIT_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-effect-geometry-audit}"
geometry_root="${GEOMETRY_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-effect-geometry}"
control_root="${CONTROL_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export PYTHONPATH="$audit_root:$audit_root/leworldmodel${PYTHONPATH:+:$PYTHONPATH}"
cd "$audit_root/leworldmodel"

"$python_bin" additional_files/diagnose_effect_geometry.py \
  --geometry-metrics "$geometry_root/leworldmodel/results/effect-geometry-full-seed0/effect_geometry_pred03_seed0/metrics.jsonl" \
  --control-metrics "$control_root/leworldmodel/results/masked-pred03-full-seed0/masked_sequence_pred03_seed0/metrics.jsonl" \
  --bank "$geometry_root/leworldmodel/results/effect-geometry-bank/bank.npz" \
  --out "$geometry_root/leworldmodel/results/effect-geometry-diagnostic.json"
