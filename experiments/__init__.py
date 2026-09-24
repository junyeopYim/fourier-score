"""Experiment orchestration: planning, scheduling, protocols and reports.

Layering rule: ``experiments`` may import ``fourier_score``; ``fourier_score``
never imports ``experiments`` (or ``scripts``/``tests``).

Orchestration lives outside ``fourier_score`` on purpose: ``source_sha256``
covers numerics only, so editing a runner, a table or a figure never changes
the checkpoint signature of a training run.  Orchestration code is
fingerprinted separately as ``orchestration_sha256`` (every
``experiments/**/*.py`` plus the ``scripts/`` entry point) in each output's
``protocol.json`` and ``execution_history.json``; see ``experiments/README.md``.
"""
