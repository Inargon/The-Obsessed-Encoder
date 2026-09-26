#!/bin/bash
set -euo pipefail

# Fixed epoch-10 evaluation for the two missing clean cells, plus a small
# evaluation-seed audit of the existing Cube result.  Run from the repo root.

repo=/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned
python_bin=/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python
out="$repo/leworldmodel/results/clean-fixed-eval-20260926"
mkdir -p "$out"

job=$(sbatch --parsable \
  --partition=interactive \
  --nodelist=SPGL-1-1 \
  --array=0-5%2 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=48G \
  --time=01:00:00 \
  --job-name=oe-clean-fixed \
  --output="$out/job-%A_%a.out" \
  <<'SBATCH'
#!/bin/bash
set -euo pipefail

repo=/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned
python_bin=/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python
out="$repo/leworldmodel/results/clean-fixed-eval-20260926"

export STABLEWM_HOME=/grp01/ids_compcog/song/swm
export LOCAL_DATASET_DIR=/grp01/ids_compcog/song/swm
export PYTHONPATH="$repo:$repo/leworldmodel"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

tasks=(pusht tworoom cube cube cube cube)
runs=(
  pusht_clean_aligned_seed0
  tworoom_clean_aligned_seed0
  cube_clean_aligned_full_seed0
  cube_clean_aligned_full_seed0
  cube_clean_aligned_full_seed0
  cube_clean_aligned_full_seed0
)
seeds=(42 42 17 42 73 137)

i="$SLURM_ARRAY_TASK_ID"
task="${tasks[$i]}"
run="${runs[$i]}"
seed="${seeds[$i]}"

cd "$repo/leworldmodel"

"$python_bin" additional_files/evaluate_clean_checkpoint.py \
  --task "$task" \
  --run-name "$run" \
  --checkpoint weights_epoch_10.pt \
  --num-eval 50 \
  --seed "$seed" \
  --output "$out/${task}-seed${seed}.json"

echo "CLEAN_FIXED_EVAL_COMPLETE task=$task seed=$seed"
SBATCH
)

echo "CLEAN_FIXED_EVAL_JOB=$job"
echo "RESULT_ROOT=$out"
squeue -j "$job"
