#!/usr/bin/env bash
#
# Reproduce every result in the README from a clean checkout.
#
#   bash run_all.sh
#
# Steps: fetch METABRIC -> cross-validated training -> figures and tables.
# Total runtime is roughly 15 minutes on four CPU cores; no GPU is needed.
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"

echo "==> [1/3] Fetching METABRIC"
bash data/download.sh

echo "==> [2/3] Cross-validated training"
"${PYTHON}" -m src.train

echo "==> [3/3] Figures and result tables"
"${PYTHON}" -m src.visualize

echo
echo "Done. Results are in:"
echo "  results/figures/   publication figures"
echo "  results/tables/    performance metrics, gene ranking"
echo "  logs/training.log  full training log"
echo
echo "To confirm a fresh run still matches the committed numbers:"
echo "  python check_reproducibility.py"
