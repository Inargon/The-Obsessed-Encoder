# Mechanism campaign — 2026-09-20

Prepared locally. NO CLUSTER JOB HAS BEEN SUBMITTED BY THIS TASK.

The cluster connection is unavailable in this desktop session. Transfer the
new files preserving repository-relative paths, then run on the cluster:

```bash
cd /grp01/ids_compcog/song/code/The-Obsessed-Encoder-decision-aligned
python_bin=/grp01/ids_compcog/song/envs/obsessed-encoder-py312/bin/python
"$python_bin" leworldmodel/additional_files/mechanism_campaign.py
"$python_bin" leworldmodel/additional_files/mechanism_campaign.py --submit --partition gpu_shared --seeds 0
```

The first command prints the plan only. The second submits seven jobs: one
smoke job, three matched training jobs (Full, joint, scalar), and three frozen
diagnostic jobs (sample seeds 17/73/137). Every formal job depends on successful
smoke. The smoke checks each training arm for two steps plus the native-pixel
intervention and model-cost path for all four existing checkpoints.

Run `sinfo -h -o '%P'` to select the exact partition name if needed. The script
validates it before submitting. The three training jobs request 20 hours each;
actual completion depends on cluster speed. Each uses one GPU. Set --seeds
0,1,2 only when intentionally launching nine training jobs. Seeds are explicit
values, unlike the legacy runner's seed-count convention.

Old joint/scalar epoch-10 checkpoint presence is printed before submission.
The new campaign uses fresh names for a matched same-source comparison; it
does not assume older checkpoints share the protocol. It never clears old
checkpoints or resumes partial runs. A worker rejects an existing target.

Job IDs and source hash are saved incrementally in
`leworldmodel/results/mechanism-<timestamp>/manifest.json`; logs are in the same
directory. If source/config changes after submission, queued workers fail
explicitly rather than silently train different code. Finish this campaign
before editing its training source, or submit from an isolated checkout.

Frozen diagnostic JSON includes checkpoint hashes, sampled indices, candidate
sources, tag colors, per-clip measurements, and bootstrap intervals. These are
sample seeds, not independent training runs. Stamping happens before resizing
and normalization, as in training. All branches use the same empirical action
candidate bank and fixed history actions. It uses five autoregressive future
steps by default and reports offline cost/selection stability, not CEM or SR.

After training, perform the common final-checkpoint SR and frozen intervention
on the new campaign checkpoint names. Do not choose best checkpoints using the
test SR. This post-training evaluation is not yet auto-submitted: the current
training callback provides periodic SR, and the published comparison must use
one verified shared evaluator. The new raw per-clip output permits paired
analysis; episode-level resampling is needed for independent-episode inference.

The theory note `theory-working-note.md` records the conditional proofs and
explicitly withdraws stronger claims from the preliminary webpage.
