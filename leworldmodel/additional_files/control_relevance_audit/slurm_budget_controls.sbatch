#!/usr/bin/env bash
#SBATCH --job-name=oe-budget-controls
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
"$python_bin" -m unittest discover -s "$audit" -p 'test_budget*.py' -v
git rev-parse HEAD
for seed in 73 74 75; do
  for mode in geometry current history; do
    "$python_bin" -u "$audit/train_budget_reachability.py" \
      --dataset "$STABLEWM_HOME/datasets/pusht_expert_train.h5" \
      --reference results/budget-reachability-114676 \
      --output "results/budget-controls-${SLURM_JOB_ID}/${mode}-${seed}" \
      --seed 73 --model-seed "$seed" --feature-mode "$mode" --epochs 200 --samples 12000
  done
done
echo "BUDGET_CONTROLS_COMPLETE results/budget-controls-${SLURM_JOB_ID}"
