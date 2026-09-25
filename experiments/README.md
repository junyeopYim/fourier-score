# experiments

Orchestration of research experiments: what to run, in which order, which
protocol is recorded, and how results become tables and figures. Numerics
(models, losses, gates, banks, training) live in `fourier_score/`.

**Layering rule.** `experiments` may import `fourier_score`; `fourier_score`
never imports `experiments`. The package is outside `source_sha256` on purpose:
editing a runner, a table or a figure does not change any checkpoint signature.
Instead every run records `orchestration_sha256` (below).

| module | role |
|---|---|
| `__main__.py` | command line: `list`, `gmm NAME ...` |
| `common.py` | provenance, protocol resume rule, execution history, CSV, figures, spawned workers |
| `gmm/registry.py` | arm suites, presets, figure styles, one spec per experiment |
| `gmm/pipeline.py` | train or audit, verify pairing, select, test; stage `all` / `report` |
| `gmm/report.py` | CSV/JSON tables, comparison and diagnostic figures |

## Commands

```bash
python -m experiments list
python -m experiments gmm NAME [flags ...] [--stage all|report]
```

`NAME` is one of `gated`, `plateau`, `spectral`, `gate_shapes`, `log_gates`.
The flags (names, defaults, validation messages) are those of the epoch-0
runners, which remain as stubs: `scripts/run_gmm_comparison.py` is
`gmm gated`, `run_gmm_plateau.py` is `gmm plateau`, `run_gmm_spectral_gate.py`
is `gmm spectral`, `run_gmm_gate_shapes.py` is `gmm gate_shapes` and
`run_gmm_log_gates.py` is `gmm log_gates`.

Stages:

- `all` (default): write or resume `OUTPUT/protocol.json`, train the new arms,
  audit reused controls (`--reuse-baselines`), select (gated only; the
  selection is written before any test observation exists), test every arm,
  then write the report (`--report`, default `OUTPUT/report`).
- `report`: rebuild the report of a finished run from its checkpoints and test
  JSON into `--report`, which is required and must lie outside `OUTPUT`.
  Nothing is trained and nothing under `OUTPUT` is written; reused
  controls are re-audited, which only reads them. Pass the flags of the
  original run: the protocol must match except for provenance. Runs recorded
  before `GMMArm.delta` existed (the archived gated and plateau runs) do not
  match and cannot be rebuilt this way.

Resume rule (`common.open_protocol`, all experiments): an existing
`protocol.json` must equal the new one except `provenance.git_revision`,
`provenance.git_dirty` and `provenance.orchestration_sha256`, which keep
their recorded values; otherwise the run stops with `Existing <label>
protocol differs; choose a new output directory`. So an interrupted run
resumes after orchestration edits (a runner, a table, a figure, another
experiment's spec) and new commits, since checkpoint signatures hold numerics
provenance only, but not after a change of flags, arms, `source_sha256`,
the python/torch/numpy versions or an `input_sha256` file. Each invocation's
own provenance, including its `orchestration_sha256`, is in
`execution_history.json`; every invocation re-runs all test evaluations and
rewrites the report. Output directories of the epoch-0 runners
(`provenance.runner_sha256`) are refused with a message naming them: resume
or re-evaluate those from the `pre-template-refactor` tree (`../fs-epoch0`).

## Provenance

`protocol.json` keeps its epoch-0 structure except `provenance`:

```
provenance = {python, torch, numpy, source_sha256,
              orchestration_sha256: {"experiments/...py": sha256, ..., "scripts/run_gmm_<x>.py": sha256},
              input_sha256: {"assets/log_gate_design/design.json": sha256},   # log_gates only
              git_revision, git_dirty}
```

Epoch-0 keys: `runner_sha256` and every `*_helpers_sha256` became
`orchestration_sha256` (every `experiments/**/*.py` plus the stub);
`design_proposal_sha256` became `input_sha256["assets/log_gate_design/design.json"]`.
The stub of an experiment is fingerprinted whether or not it was the invoking
command.

- **Checkpoints.** The provenance given to `train_arm`, hence stored in each
  `checkpoint.pt` and folded into its signature, is `{python, torch, numpy,
  source_sha256}` only (epoch-0 runs stored the whole runner provenance).
- **`execution_history.json`** gets one entry per invocation with that
  invocation's full provenance; `--stage report` appends its entry
  (`"stage": "report"`) only to the copy in the report directory.
- **`checkpoint_audit.json`.** Rows of reused checkpoints add
  `origin_output_root` and `origin_protocol_sha256` (sha256 of that root's
  `protocol.json`), since the reused checkpoint's signature no longer names
  the runner that trained it. Reuse itself is audited bit-exactly (config,
  initial and final EMA digests, every validation noise bin), not by signature.
- **Figures.** `common.save_figure` writes PNG, SVG and PDF with a fixed SVG id
  salt and SVG/PDF dates from `SOURCE_DATE_EPOCH` (default 0), so rebuilt
  figures are byte-identical.

## Adding a GMM gate experiment

1. **Gate (numerics, changes `source_sha256`).** Implement the mode in
   `fourier_score` (a `GATES` row in `fourier_score/gates.py` and its weights
   in `FourierGaussian._gate_weights`) with its tests. Its arms are
   `gate_arm(covariance, mode, **params)` from `fourier_score/gmm.py`; no new
   factory is needed. Nothing in `experiments/` is needed for this step.
2. **Suite.** In `gmm/registry.py` append
   `SUITES["<name>"] = (*SUITES["log_gates"], *new default arms)`; the new
   experiment's controls are the previous suite, in order.
3. **Style.** Add `STYLES["<gate_mode>"] = (label, (scalar colour, Fourier
   colour), (scalar style, Fourier style))`. If two gates drawn in one figure
   share a Fourier colour, `ALTERNATE_COLORS` gives one of them another.
4. **Spec.** Add a `FixedGateExperiment` last in `EXPERIMENTS`. Required:
   `name`, `description` (`--help` text), `report` (the write-up in
   `reports/gmm/`), `figure_stem`, `options` (flags), `new_modes`,
   `new_arms(args)`, `controls(args)` (usually `lambda args: SUITES["log_gates"]`),
   a new `test_bank` (earlier test banks and `gmm-oracle-v1` are rejected
   automatically), `criterion`, `check(cfg, args)`, `transition(cfg, args)`,
   `fixed` and `diagnostics` (`Diagnostics(layout, figure stem, gate modes)`).
   Usually `extras` (gate definitions, constraints) too. The defaults are
   the modern ones: several `--reuse-baselines` roots, 16 workers,
   identical parameter counts asserted, `label = name`, and no `stub`, which
   only the five legacy experiments have (`python -m experiments gmm <name>`
   is the entry point).
5. **Run** into `saved/`, reusing every earlier output:
   `python -m experiments gmm <name> --output saved/gmm_<name>_<date>
   --reuse-baselines saved/gmm_gated_... saved/gmm_log_gates_...`.

## Assets rule

For a new experiment commit to `assets/gmm_<name>_comparison/` only the figures
(PNG, plus SVG where a vector copy is useful) and `summary.csv` /
`paired_comparisons.csv`. Keep the bulky outputs (`checkpoint_audit.json`,
per-seed, noise- and frequency-resolved CSVs, PDFs, checkpoints) in `saved/`
and record their sha256 in the report under `reports/gmm/`. Earlier
`assets/gmm_*` directories are records and are never rewritten.
