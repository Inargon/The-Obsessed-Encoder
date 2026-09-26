#!/bin/bash
set -euo pipefail

# Submit one protocol-matched clean PushT JEPA campaign:
#   smoke -> 10-epoch training -> fixed seed-42 evaluation.
# The final array evaluates both epoch 10 and step 70k, matching the two Ours
# checkpoints already reported locally (82% and 86%, respectively).

repo="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
campaign="$repo/leworldmodel/results/pusht-clean-jepa-matched-20260926"
mkdir -p "$campaign"

# Fail before submitting GPU work if the supposedly pure JEPA arm accidentally
# acquires a tag, control loss, or router in a future edit.
cd "$repo"
PYTHONPATH="$repo" "$python_bin" -m pytest \
  leworldmodel/additional_files/tests/test_pusht_clean_jepa_matched.py -q

common_export="ALL,REPO_ROOT=$repo,OBSESSED_PYTHON=$python_bin"

smoke_job=$(sbatch --parsable \
  --partition=gpu_shared \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=00:20:00 \
  --job-name=oe-pt-cj-smoke \
  --output="$campaign/smoke-%j.out" \
  --export="$common_export,PUSHT_CLEAN_JEPA_ARM=pusht_clean_jepa_matched_smoke" \
  --wrap="
    set -euo pipefail
    export STABLEWM_HOME=/grp01/ids_compcog/song/swm
    export LOCAL_DATASET_DIR=/grp01/ids_compcog/song/swm
    export SPT_CACHE_DIR=/grp01/ids_compcog/song/cache/stable-pretraining
    export TMPDIR=/grp01/ids_compcog/song/tmp/aluo/\${SLURM_JOB_ID}
    export WANDB_MODE=disabled
    export PYTHONPATH=$repo:$repo/leworldmodel
    mkdir -p \"\$SPT_CACHE_DIR\" \"\$TMPDIR\"
    cd $repo/leworldmodel
    $python_bin additional_files/run_pusht_clean_jepa_matched.py \
      --seeds 1 --gpus 0 --results-dir $campaign/smoke \
      --extra-opts '+trainer.max_steps=20 +trainer.log_every_n_steps=1 ++eval.every_n_steps=1000000 ++checkpoint.every_n_steps=1000000'
    echo PUSHT_CLEAN_JEPA_MATCHED_SMOKE_COMPLETE
  ")

train_job=$(sbatch --parsable \
  --partition=gpu_shared \
  --dependency="afterok:$smoke_job" \
  --kill-on-invalid-dep=yes \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=20:00:00 \
  --job-name=oe-pt-cj-train \
  --output="$campaign/train-%j.out" \
  --export="$common_export,PUSHT_CLEAN_JEPA_ARM=pusht_clean_jepa_matched_full" \
  --wrap="
    set -euo pipefail
    export STABLEWM_HOME=/grp01/ids_compcog/song/swm
    export LOCAL_DATASET_DIR=/grp01/ids_compcog/song/swm
    export SPT_CACHE_DIR=/grp01/ids_compcog/song/cache/stable-pretraining
    export TMPDIR=/grp01/ids_compcog/song/tmp/aluo/\${SLURM_JOB_ID}
    export WANDB_MODE=disabled
    export PYTHONPATH=$repo:$repo/leworldmodel
    mkdir -p \"\$SPT_CACHE_DIR\" \"\$TMPDIR\"
    cd $repo/leworldmodel
    $python_bin additional_files/run_pusht_clean_jepa_matched.py \
      --seeds 1 --gpus 0 --results-dir $campaign/train \
      --extra-opts '++checkpoint.every_n_steps=5000'
    echo PUSHT_CLEAN_JEPA_MATCHED_TRAIN_COMPLETE
  ")

eval_job=$(sbatch --parsable \
  --partition=interactive \
  --nodelist=SPGL-1-1 \
  --dependency="afterok:$train_job" \
  --kill-on-invalid-dep=yes \
  --array=0-1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=48G \
  --time=01:00:00 \
  --job-name=oe-pt-cj-eval \
  --output="$campaign/eval-%A_%a.out" \
  --export="$common_export,CAMPAIGN=$campaign" \
  <<'SBATCH'
#!/bin/bash
set -euo pipefail

export STABLEWM_HOME=/grp01/ids_compcog/song/swm
export LOCAL_DATASET_DIR=/grp01/ids_compcog/song/swm
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/leworldmodel"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

checkpoints=(weights_epoch_10.pt weights_step_70000.pt)
labels=(epoch10 step70000)
i="$SLURM_ARRAY_TASK_ID"

cd "$REPO_ROOT/leworldmodel"
"$OBSESSED_PYTHON" additional_files/evaluate_clean_checkpoint.py \
  --task pusht \
  --run-name pusht_clean_jepa_matched_full_seed0 \
  --checkpoint "${checkpoints[$i]}" \
  --num-eval 50 \
  --seed 42 \
  --output "$CAMPAIGN/${labels[$i]}-seed42.json"

echo "PUSHT_CLEAN_JEPA_MATCHED_EVAL_COMPLETE checkpoint=${labels[$i]}"
SBATCH
)

cat > "$campaign/manifest.json" <<EOF
{
  "benchmark": "clean PushT",
  "dataset": "pusht_expert_train.h5",
  "method": "upstream LeWM prediction + SIGReg (no L_ctrl, no routing)",
  "training_seed": 0,
  "epochs": 10,
  "checkpoint_interval_steps": 5000,
  "evaluation_seed": 42,
  "evaluation_episodes": 50,
  "comparison_checkpoints": ["weights_epoch_10.pt", "weights_step_70000.pt"],
  "smoke_job": "$smoke_job",
  "train_job": "$train_job",
  "eval_job": "$eval_job"
}
EOF

echo "SMOKE_JOB=$smoke_job"
echo "TRAIN_JOB=$train_job"
echo "EVAL_JOB=$eval_job"
echo "CAMPAIGN=$campaign"
squeue -j "$smoke_job,$train_job,$eval_job"
