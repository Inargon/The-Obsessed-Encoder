#!/usr/bin/env bash
#SBATCH --job-name=oe-audit-smoke
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
repo_root="${REPO_ROOT:-/grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned}"
python_bin="${OBSESSED_PYTHON:-/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python}"
export STABLEWM_HOME="${STABLEWM_HOME:-/grp01/ids_compcog/song/swm}"
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-/grp01/ids_compcog/song/swm}"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root/leworldmodel"
audit=additional_files/control_relevance_audit
out="$repo_root/leworldmodel/results/control-relevance-smoke-${SLURM_JOB_ID:?Submit through sbatch}"
mkdir "$out"
"$python_bin" -m unittest discover -s "$audit" -p test_audit.py -v
for label in jepa ours; do
 if [ "$label" = jepa ]; then run=pusht_clean_jepa_matched_full_seed0; else run=pusht_clean_ours_matched_full_seed0; fi
 "$python_bin" -u "$audit/export_features.py" --run "$run" --checkpoint weights_epoch_10.pt --dataset "$STABLEWM_HOME/datasets/pusht_expert_train.h5" --cache-dir "$LOCAL_DATASET_DIR" --samples 128 --horizon 5 --seed 73 --out "$out/$label.npz"
 "$python_bin" -u "$audit/audit.py" --input "$out/$label.npz" --out "$out/$label-report.json" --seed 73
done
"$python_bin" - "$out" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
with np.load(root/'jepa.npz',allow_pickle=False) as a, np.load(root/'ours.npz',allow_pickle=False) as b:
 for key in ['row_index','episode_id','state','previous_state','next_state','action']:
  np.testing.assert_array_equal(a[key],b[key],err_msg='Unmatched input '+key)
reports={name:json.loads((root/(name+'-report.json')).read_text()) for name in ['jepa','ours']}
assert reports['jepa']['protocol']['episodes']==reports['ours']['protocol']['episodes']
print('MATCHED_ROWS_AND_EPISODE_SPLITS_VERIFIED')
for name,report in reports.items():
 print('MODEL',name,'COUNTS',report['protocol']['counts'])
 for key,value in report['state_probes'].items():print(key,'R2',value['r2'])
 for key,value in report['outcome_probes'].items():print(key,{k:v['r2'] for k,v in value.items()})
print('RESULT_DIR',root)
print('AUDIT_SMOKE_COMPLETE')
PY
