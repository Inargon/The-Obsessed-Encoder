# Verifying that there are no unreported changes

The three examples embed published code, and the claim throughout this
repository is: **every modification lives inside a marked
`# >>> obsessed-encoder ... # <<< obsessed-encoder` block, and
everything else is byte-identical to the published upstream.** This page is
the audit trail for that claim — a handful of git commands you can run
yourself, with the expected outputs recorded (as of this repository's current
history and the pinned upstream commits). Line counts below are
`git diff --no-index --numstat` values (added/deleted lines, blank lines
included).

Directories exempt from the claim, because they are this repository's own
code and say so: `common/`, `tools/`, and each example's `additional_files/`
(inventoried in each example's `additional_files/README.md`).

Setup used by all three sections:

```bash
git clone https://github.com/Enigma-Incorporated/The-Obsessed-Encoder.git
cd The-Obsessed-Encoder
```

---

## 1. DINOv3 — vendored `facebookresearch/dinov3`

Clone the original at the pinned commit:

```bash
git clone https://github.com/facebookresearch/dinov3 /tmp/upstream-dinov3
git -C /tmp/upstream-dinov3 checkout 346f38fee679c56a6888f91c51670fae61d364e0
```

Show every file that differs (our additions excluded by name):

```bash
diff -qr /tmp/upstream-dinov3 dinov3 -x .git -x additional_files
```

Expected output — three modified files, nothing else:

```
Files /tmp/upstream-dinov3/dinov3/data/loaders.py and dinov3/dinov3/data/loaders.py differ
Files /tmp/upstream-dinov3/dinov3/train/ssl_meta_arch.py and dinov3/dinov3/train/ssl_meta_arch.py differ
Files /tmp/upstream-dinov3/dinov3/train/train.py and dinov3/dinov3/train/train.py differ
Only in /tmp/upstream-dinov3: notebooks
```

(`notebooks/` is upstream's demo-notebook directory, omitted from the vendored
copy rather than modified; nothing in the training path references it.)

Per-file diffs:

```bash
git diff --no-index /tmp/upstream-dinov3/dinov3/data/loaders.py      dinov3/dinov3/data/loaders.py
git diff --no-index /tmp/upstream-dinov3/dinov3/train/ssl_meta_arch.py dinov3/dinov3/train/ssl_meta_arch.py
git diff --no-index /tmp/upstream-dinov3/dinov3/train/train.py       dinov3/dinov3/train/train.py
```

What to look for: `loaders.py` is +10/−0 and `ssl_meta_arch.py` +11/−0 — pure
insertions, fully enclosed in sentinel blocks. `train.py` is +76/−1; the one
removed line is an identifier substitution (`collate_data_and_cast` →
`collate_base`) whose surrounding sentinel block, in the same hunk, defines
`collate_base = collate_data_and_cast` when the watermark is off. The intent
of each block is inventoried in `dinov3/additional_files/README.md`.

## 2. LeWorldModel — vendored `lucas-maes/le-wm`

```bash
git clone https://github.com/lucas-maes/le-wm /tmp/upstream-lewm
git -C /tmp/upstream-lewm checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
```

```bash
diff -qr /tmp/upstream-lewm leworldmodel -x .git -x additional_files
```

Expected output — one modified file plus a convenience copy of its diff:

```
Only in leworldmodel: changes.diff
Files /tmp/upstream-lewm/train.py and leworldmodel/train.py differ
```

Per-file diff:

```bash
git diff --no-index /tmp/upstream-lewm/train.py leworldmodel/train.py
```

(`leworldmodel/changes.diff` is a committed copy of this same diff with a
comment header; the `index 84794e9..18a4662` blob hashes in your live output
should match the ones recorded there.)

What to look for: +41/−2. The two removed lines are argument substitutions
inside the marked trainer-construction hunk (`callbacks=[object_dump_callback]`
gains the opt-in callback list; `logger` becomes the wrapped `trainer_logger`)
— with the corresponding configs absent, both evaluate to exactly the
upstream values.

## 3. LeJEPA — derived from the upstream minimal recipe

The upstream here is not a standalone `.py` file: the published minimal
recipe lives as the python code blocks of
[`MINIMAL.md`](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md).
Extract them and diff against our single training file:

```bash
git clone https://github.com/galilai-group/lejepa /tmp/upstream-lejepa
git -C /tmp/upstream-lejepa checkout c293d291ca87cd4fddee9d3fffe4e914c7272052

awk '/^```python/{f=1;next} /^```/{if(f){f=0;print ""}next} f' \
    /tmp/upstream-lejepa/MINIMAL.md > /tmp/minimal_extracted.py

git diff --no-index /tmp/minimal_extracted.py lejepa/lejepa_minimal.py
```

Expected shape: +311/−25 — small enough to read end to end, and that is the
intended audit here. Every insertion is either (a) inside a sentinel block,
(b) the module docstring, or (c) one of the two throughput-only deviations the
docstring declares up front (configurable DataLoader workers with
pin_memory, and loading the same `frgfm/imagenette` images via the
auto-converted parquet revision instead of the retired loading script). Of the
25 removed lines, most reappear inside the default branch of a sentinel block
in the same hunk (verbatim, or as a default-equivalent parametrization -- e.g.
the upstream dataset load becomes the `else:` arm of the watermark seam, and
hardcoded literals become parameters whose defaults are those literals); the
rest belong to deviation (a) or were upstream's Imagenette/MNIST-specific
usage lines, which have no counterpart here.
The per-change inventory is in `lejepa/additional_files/README.md`.

---

## Why the marked blocks are inert by default

Every sentinel block is inert at its defaults: with the experiment's env vars
and config keys absent, the code path is upstream's, and the examples' test
suites assert exactly that. Two precision notes for a careful reader. First,
the DINOv3 dataset registration (`loaders.py`) is *activated* by the shipped
config's `dataset_path: ImageNet1kParquet:...` — the block exists so the
vendored trainer can read the pinned local ImageNet-1k shards at all; it
selects data, and its rendering seam stays inert unless the `DINOV3_WM_*` env
vars are set. Second, the shipped arms deliberately run with the measurement
env (`DINOV3_PROBE=1`, metrics mirrors, pair suites on watermarked arms)
switched ON — "inert by default" describes the un-configured path a public
user gets, not the shipped experiment, whose every activation is inventoried
in the per-example `additional_files/README.md`.
