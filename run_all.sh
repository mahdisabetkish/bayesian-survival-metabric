#!/usr/bin/env bash
#
# Reproduce every result in the README from a clean checkout.
#
#   bash run_all.sh
#
# Steps: fetch METABRIC -> cross-validated training -> CoxNet baseline -> figures and tables,
# then the network and CoxNet again on relapse-free survival.
# Total runtime is roughly 40 minutes on four CPU cores; no GPU is needed.
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"

echo "==> [1/6] Fetching METABRIC"
bash data/download.sh

echo "==> [2/6] Cross-validated training (overall survival)"
"${PYTHON}" -m src.train

echo "==> [3/6] CoxNet baseline on the same folds"
"${PYTHON}" -m src.coxnet_baseline

echo "==> [4/6] Figures and result tables"
"${PYTHON}" -m src.visualize

echo "==> [5/6] Cross-validated training (relapse-free survival)"
"${PYTHON}" -m src.train --endpoint rfs

echo "==> [6/6] CoxNet baseline on relapse-free survival"
"${PYTHON}" -m src.coxnet_baseline --endpoint rfs

echo
echo "Done. Results are in:"
echo "  results/figures/   publication figures"
echo "  results/tables/    performance metrics, gene ranking"
echo "  results/rfs/       the same three models on relapse-free survival"
echo "  logs/training.log  full training log"
echo
echo "To confirm a fresh run still matches the committed numbers:"
echo "  python check_reproducibility.py"
