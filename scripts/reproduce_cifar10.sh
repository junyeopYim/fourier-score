#!/usr/bin/env bash
# Run from the repository root. Add --dry-run to inspect without downloading/training.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  printf 'Usage: bash scripts/reproduce_cifar10.sh {50k|950k|1m3} [comparison options]\n' >&2
  exit 2
fi

case "$1" in
  50k) config=configs/cifar10_ablation.json ;;
  950k) config=configs/cifar10_950k.json ;;
  1m3) config=configs/cifar10_1m3.json ;;
  *) printf 'Unknown budget: %s (choose 50k, 950k, or 1m3)\n' "$1" >&2; exit 2 ;;
esac
shift

exec uv run --locked python scripts/run_comparison.py -c "$config" "$@"
