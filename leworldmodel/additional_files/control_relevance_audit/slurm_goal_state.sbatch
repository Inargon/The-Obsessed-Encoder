#!/usr/bin/env bash
#SBATCH --job-name=oe-goal-state
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
out="results/goal-state-${SLURM_JOB_ID}"
data=results/goal-role-114739/data
"$python_bin" -u "$audit/recover_goal_role_state.py" --data "$data" --output "$out/states.npz"
"$python_bin" -u "$audit/train_goal_role.py" --features "$data/features.npz" --state-cache "$out/states.npz" --output "$out/state"
"$python_bin" -u "$audit/train_goal_role.py" --features "$data/features.npz" --output "$out/visual"
echo "GOAL_STATE_COMPARISON_COMPLETE $out"
