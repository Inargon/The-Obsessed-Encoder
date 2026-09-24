# Experiment data registry

This directory stores measured experiment data without interpretation.
Arguments, claims, and recommendations belong in the article, not here.

## Files

- `protocols.json`: evaluation and diagnostic protocol definitions.
- `pusht-core4-epoch10.json`: matched JEPA, Joint, Scalar, and Ours data.
- `pusht-lctrl-ablation-epoch10.json`: IDM-only, Inverse+Cycle,
  Inverse+Reach, and Full control-objective ablations.
- `cross-task-clean.json`: clean Cube, TwoRoom, and Reacher measurements.
- `additional-runs.json`: gradient audit, counterfactual-planning run, and
  Decision-Subspace prototype.
- `catalog.json`: registry index and update history.
- `validate_registry.py`: standard-library validation and long-form CSV export.

## Data conventions

- Success rates are stored as fractions in `[0, 1]`.
- A `{ "mean": x, "std": y }` object reports the mean and standard deviation
  over diagnostic sampling seeds, not training-seed uncertainty.
- `training_seed` identifies model training initialization.
- `evaluation_seed` identifies fixed environment evaluation sampling.
- `diagnostic_sampling_seeds` vary frozen-probe or diagnostic sampling only.
- Online training-curve measurements and standalone fixed-checkpoint
  evaluations are stored under different protocol IDs.
- Missing values are omitted. They are never written as zero.
- Every record includes a provenance field. `cluster_output_transcription`
  means the value was copied from preserved terminal output in the research
  thread and should later be linked to an archived raw result file when one is
  available.

## Validation and CSV export

From the repository root:

```bash
python article/experiment-data/validate_registry.py
python article/experiment-data/validate_registry.py \
  --csv article/experiment-data/metrics-long.csv
```

The CSV is derived output. JSON files are the source of truth.

## Local preview

From the repository root, run:

```powershell
$env:PORT = 8766
node article/serve-preview.mjs
```

Then open `http://127.0.0.1:8766/`. The preview is read-only and loads the
JSON source files without caching.
