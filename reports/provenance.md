# Provenance

[Back to README](../README.md#repository-layout) · [All reports](README.md)

Every pixel and latent checkpoint and every GMM runner study records
`source_sha256`: a SHA-256 over the relative path and bytes of every `*.py` file
below `fourier_score/`, taken in sorted order. Only the numerical package enters
this hash. Documentation, `reports/`, `experiments/`, `scripts/`, tests and
configs do not, so they can be reorganized without affecting existing runs.
A **provenance epoch** is the history up to a Git tag. Runs recorded under the
tag's hash can be resumed in a worktree at that tag.

## Epochs

| Epoch | Git tag | Commit | `source_sha256` at the tag | Covers |
|---|---|---|---|---|
| 0 | `pre-template-refactor` | `48325d3` | `f79717f3da852ffcde3ab8eea4997b3b44a3d0843e6cea040bde9afa145970a8` | All runs, checkpoints and `assets/` results produced before the template refactor, including every study in these reports. The package changed during this history, so the earlier studies recorded [other hashes](#source-of-each-archived-study) |
| 1 | `epoch-1` | `8e76031` | `d6d0c939e3a1c565aa25f0b520da7644c8643834568297ca83f55661a656502d` | Runs after the pytorch-template layout of `fourier_score/` (`model/`, `data_loader/`, `trainer/`, `gates.py`, `parse_config.py`, `provenance.py`). Numerics match epoch 0 bit for bit on the golden contracts and differential checks; epoch-0 checkpoints load here with a warning and resume only at `pre-template-refactor` |

Later epochs add a row when a change to `fourier_score/` is merged. To print the
hash of the current checkout:

```bash
uv run --locked python -c "from fourier_score.provenance import source_hash; print(source_hash())"
```

## What the hash guards

- **Resume is refused across source hashes by design.** `train.py -r`, the latent
  trainer, `scripts/run_mnist_remaining.py` and the GMM runners stop when a
  checkpoint's recorded `source_sha256` differs from the checkout, so one run
  never mixes numerics from two sources. GMM runners store the hash in
  `protocol.json` and every per-arm checkpoint signature.
- **Inference warns but proceeds.** `sample.py`, `evaluate.py dsm` and the
  latent loaders accept an older checkpoint and warn that the output is a new
  evaluation protocol. Rerun such evaluations at the matching source when the
  numbers must be comparable with archived results.
- The GMM notebook records hashes of the package files it uses and
  `notebook_code_sha256`; editing either prevents continuing an earlier run tag
  under `saved/gmm_oracle/`.

## Resume or re-evaluate an epoch-0 run

Create a separate worktree at the tag and run the unchanged commands there:

```bash
# From the main checkout, once:
git worktree add ../fs-epoch0 pre-template-refactor
cd ../fs-epoch0
uv sync --locked --all-extras
uv run --no-sync python -c "from fourier_score.utils import source_hash; print(source_hash())"
# Expected: f79717f3da852ffcde3ab8eea4997b3b44a3d0843e6cea040bde9afa145970a8
```

The new worktree has no `data/`, `saved/` or `pretrained/`. Point the commands
at the main checkout with absolute paths. `trainer.save_dir`, the data root and
the statistics cache are outside the resume signature; inference accepts only
the data root (statistics come from the checkpoint):

```bash
MAIN=$(realpath ../fourier-score)   # the main checkout that holds saved/ and data/
uv run --no-sync python train.py -r "$MAIN/saved/<run>/last.pt" \
  --set trainer.save_dir="$MAIN/saved" \
  --set data_loader.args.root="$MAIN/data" --set fourier.cache_dir="$MAIN/data/stats"
uv run --no-sync python evaluate.py dsm -r "$MAIN/saved/<run>/last.pt" \
  -o "$MAIN/saved/<run>_dsm_epoch0.json" --set data_loader.args.root="$MAIN/data"
```

For a GMM study recorded under the tag's hash, such as the log-axis study,
repeat the command from its report in the worktree with the same options and
`--output "$MAIN/saved/<study directory>"`. Completed arms are reused and
interrupted arms resume. An earlier study stops with a protocol mismatch
instead; see [below](#source-of-each-archived-study).
`scripts/run_mnist_remaining.py` and `scripts/run_comparison.py` are historical
runners; continue their studies from this worktree as well.
Notebook runs are continued by copying their `saved/gmm_oracle/<preset>/<tag>/`
directory into the worktree and reusing the same `GMM_RUN_TAG`.

Keep the worktree detached at the tag: do not commit to it or update it. It is
removed with `git worktree remove ../fs-epoch0` once no epoch-0 run needs to be
continued.

## Source of each archived study

Before the tag, the package changed with each GMM study. The new runs of each
runner study record the package hash of the commit that added the study:

| Study | `source_sha256` of its new runs | Same package source as |
|---|---|---|
| [Loss comparison](gmm/loss-comparison.md) | Notebook protocol: per-file hashes in `summary.json` | Recorded commit `f8df48f` |
| [Gated comparison](gmm/gated-comparison.md) | `67974920…` | `77ec064` |
| [Plateau comparison](gmm/plateau-comparison.md) | `a69a635a…` | `08b3acd` |
| [Spectral gate](gmm/spectral-gate.md) | `792457d1…` | `f83a43f` |
| [Linear and tanh gates](gmm/linear-tanh-gates.md) | `f40484c3…` | `4c9c3b5` |
| [Log-axis gates](gmm/log-axis-gates.md) | `f79717f3…` (the tag's hash) | `6c369a4` to `48325d3` |

Archived results name files by their paths at the time, such as the loss
comparison's per-file hashes. [path-map.json](path-map.json) gives the current
location of every file that the template refactor moved or split, with its
SHA-256 at the tag.

A later study reuses earlier checkpoints only after reproducing their
configuration, initialization, final EMA digest and every validation noise bin
exactly (its `checkpoint_audit.json`), not by matching hashes, so one audit can
list several hashes. The runners' protocols include `source_sha256`, so the
epoch-0 worktree cannot resume or extend a study recorded under an earlier
hash. Reuse its checkpoints from a new output directory with
`--reuse-baselines`, as the later studies did, or rerun it in a worktree at the
listed commit.
