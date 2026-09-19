"""Submit matched controls + frozen tag interventions without overwriting runs.

Run on the cluster with its training Python. Default is a printed plan;
--submit performs preflight and submits an afterok-gated campaign.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ('full','joint','scalar')
CHECKPOINTS = (
    'jepa=colored_square_episode_seed0/weights_epoch_10.pt',
    'full=control_aligned_pred1_seed0/weights_epoch_10.pt',
    'parallel=control_parallel_pred1_seed0/weights_epoch_10.pt',
    'gated_orth=pusht_tagged_orthogonal_only_gated_pred1_seed0/weights_epoch_10.pt',
)


def source_digest():
    digest=hashlib.sha256()
    files=list((REPO/'leworldmodel').glob('*.py'))
    for directory in (HERE, REPO/'leworldmodel/config', REPO/'common'):
        files += [p for p in directory.rglob('*') if p.suffix in ('.py','.yaml')]
    for p in sorted(set(files)):
        digest.update(str(p.relative_to(REPO)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def environment():
    env=dict(os.environ)
    env['PYTHONPATH']=str(REPO)+os.pathsep+str(REPO/'leworldmodel')
    env['WANDB_MODE']='disabled'
    env['MUJOCO_GL']='egl'
    env.pop('PYOPENGL_PLATFORM',None)
    env.setdefault('STABLEWM_HOME','/grp01/ids_compcog/song/swm')
    env.setdefault('LOCAL_DATASET_DIR',env['STABLEWM_HOME'])
    return env


def arm_spec(arm):
    from run_control import load_control_configs
    cfg=load_control_configs()
    base=dict(cfg['arms']['control_aligned_pred1'])
    overrides=[v for v in base['overrides'] if 'loss.aligned_gradient_routing.' not in v]
    overrides.append('+loss.aligned_gradient_routing.enabled='+('false' if arm=='joint' else 'true'))
    if arm=='scalar':
        overrides.append('+loss.aligned_gradient_routing.routing_mode=norm_matched_scalar')
    base['overrides']=overrides
    return cfg,base


def train(root, arm, seed, smoke):
    import run as base_runner
    cfg,spec=arm_spec(arm)
    prefix=f"{root.name}_{'smoke' if smoke else 'full'}_{arm}"
    name=f'{prefix}_seed{seed}'
    checkpoint=Path(os.environ['STABLEWM_HOME'])/'checkpoints'/name
    target=root/name
    if checkpoint.exists() or target.exists():
        raise RuntimeError(f'Refusing existing run: {name}; use a new campaign')
    options='++checkpoint.every_n_steps=5000 ++eval.every_n_steps=2000'
    if smoke:
        options='++trainer.max_steps=2 ++checkpoint.every_n_steps=1000000 ++eval.every_n_steps=1000000'
    args=argparse.Namespace(results_dir=str(root),epochs=1 if smoke else 10,
        extra_opts=options,tag=root.name)
    built=base_runner.build_spec(prefix,spec,cfg,seed,args)
    target.mkdir()
    (target/'command.json').write_text(json.dumps(built.command,indent=2))
    # Config composition is checked before allocating hours to a training run.
    with (target/'resolved-config.yaml').open('w') as f:
        subprocess.run(built.command+['--cfg','job','--resolve'],cwd=built.cwd,
                       env={**os.environ,**built.env},stdout=f,check=True)
    subprocess.run(built.command,cwd=built.cwd,env={**os.environ,**built.env},check=True)
    expected=checkpoint/('weights_epoch_1.pt' if smoke else 'weights_epoch_10.pt')
    if not expected.is_file():
        raise RuntimeError(f'Training returned success without expected checkpoint: {expected}')
    (target/'complete.json').write_text(json.dumps({'checkpoint':str(expected),'seed':seed,'arm':arm}))


def diagnose(root,seed,smoke):
    cmd=[sys.executable,str(HERE/'diagnose_tag_intervention.py'),'--seed',str(seed),
         '--num-clips','2' if smoke else '128','--candidates','2' if smoke else '32',
         '--out',str(root/f"tag-{'smoke' if smoke else 'full'}-{seed}.json")]
    for spec in CHECKPOINTS: cmd+=['--checkpoint',spec]
    subprocess.run(cmd,cwd=REPO/'leworldmodel',env=environment(),check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--seeds',default='0',help='Explicit training seeds, e.g. 0 or 0,1,2 (NOT a count)')
    p.add_argument('--partition',default='gpu_shared')
    p.add_argument('--worker',choices=('smoke','train','diagnose'))
    p.add_argument('--campaign',type=Path)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    os.environ.update(environment())
    if args.worker:
        if not args.campaign: p.error('--worker requires --campaign')
        manifest=json.loads((args.campaign/'manifest.json').read_text())
        if source_digest()!=manifest['source_sha256']:
            raise RuntimeError('Source/config changed since submission; use a fresh campaign')
        if args.worker=='smoke':
            for arm in ARMS: train(args.campaign,arm,0,True)
            diagnose(args.campaign,17,True)
        elif args.worker=='train':
            if not args.arm: p.error('train worker requires --arm')
            train(args.campaign,args.arm,args.seed,False)
        else: diagnose(args.campaign,args.seed,False)
        print(f'MECHANISM_{args.worker.upper()}_COMPLETE',flush=True)
        return
    seeds=list(dict.fromkeys(int(s) for s in args.seeds.split(',')))
    if not seeds or min(seeds)<0: p.error('Need nonnegative explicit seeds')
    print(f'Training: {ARMS}, seeds={seeds}, 10 epochs; 1 GPU per job.')
    print('Frozen diagnostic: four existing epoch-10 checkpoints, sample seeds 17/73/137.')
    print('Smoke: each training arm for 2 steps + two-clip diagnostic; formal jobs require afterok.')
    if not args.submit:
        print('PLAN ONLY. Pass --submit ON THE CLUSTER to submit. No jobs launched.')
        return
    partitions=subprocess.check_output(['sinfo','-h','-o','%P'],text=True).replace('*','').split()
    if args.partition not in partitions:
        raise SystemExit(f'Unknown partition {args.partition!r}; available: {sorted(set(partitions))}')
    ckpt=Path(os.environ['STABLEWM_HOME'])/'checkpoints'
    for spec in CHECKPOINTS:
        _,location=spec.split('=',1)
        path=ckpt/location
        if not path.is_file() or not (path.parent/'config.json').is_file():
            raise SystemExit(f'Missing checkpoint/config: {path}; nothing submitted')
    # Expose potentially reusable historical comparisons before launching new names.
    for name in ('masked_sequence_pred1_seed0','control_norm_matched_scalar_pred1_seed0'):
        print(f'HISTORICAL {name}: epoch10={(ckpt/name/"weights_epoch_10.pt").is_file()}')
    root=REPO/'leworldmodel/results'/('mechanism-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    root.mkdir(parents=True,exist_ok=False)
    manifest={'source_sha256':source_digest(),'training_seeds':seeds,'jobs':[],
              'historical_checkpoints':CHECKPOINTS,'interpretation':'norm matching at each arm own current parameters'}
    def save(): (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save()
    def submit(worker,extra,dependency=None):
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',worker,'--campaign',str(root),*extra]
        batch=['sbatch','--parsable','--partition',args.partition,'--gres=gpu:1',
               '--cpus-per-task=12','--mem=64G','--time=00:30:00' if worker=='smoke' else '--time=20:00:00' if worker=='train' else '--time=02:00:00',
               '--job-name=oe-mech-'+worker,'--chdir',str(REPO),'--output',str(root/'job-%j.out')]
        if dependency: batch+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        batch+=['--wrap',shlex.join(command)]
        job=subprocess.check_output(batch,text=True).strip().split(';')[0]
        if not job.isdigit(): raise RuntimeError(f'Unexpected sbatch response {job!r}')
        manifest['jobs'].append({'job_id':job,'worker':worker,'args':extra,'dependency':dependency})
        save()
        print(f'SUBMITTED {job} {worker} {extra}',flush=True)
        return job
    smoke=submit('smoke',[])
    for seed in seeds:
        for arm in ARMS: submit('train',['--arm',arm,'--seed',str(seed)],smoke)
    for seed in (17,73,137): submit('diagnose',['--seed',str(seed)],smoke)
    print(f'JOB_MANIFEST={root/"manifest.json"}')


if __name__=='__main__':
    main()
