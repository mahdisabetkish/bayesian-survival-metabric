"""Verify that a fresh run reproduces the committed results.

Retrains from scratch and compares the headline metrics against
``results/tables/performance_metrics.csv``. Run after changing anything in
``src/`` to confirm the numbers in the README still hold.

    python check_reproducibility.py [--tolerance 0.002]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.coxnet_baseline import comparison_table, run_coxnet_cv
from src.data_loader import load_metabric
from src.train import TrainConfig, configure_logging, run_cross_validation
from src.uncertainty import evaluate, performance_table

COMPARED_COLUMNS = ["cindex", "auc_5y"]
COXNET_ROW = "CoxNet, 1,000 genes + clinical"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/brca_metabric")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()

    reference_path = Path(args.results_dir) / "tables" / "performance_metrics.csv"
    if not reference_path.exists():
        print(f"No committed results at {reference_path}; run `bash run_all.sh` first.")
        return 2

    configure_logging(Path("logs/reproducibility.log"), verbose=False)
    reference = pd.read_csv(reference_path).set_index("model")

    print("Retraining from scratch...")
    data = load_metabric(args.data_dir)
    oof, _ = run_cross_validation(data, TrainConfig())
    fresh = performance_table(evaluate(oof)).set_index("model")
    checks = [
        (model, column, float(reference.loc[model, column]), float(fresh.loc[model, column]))
        for model in reference.index
        for column in COMPARED_COLUMNS
    ]

    comparison_path = Path(args.results_dir) / "tables" / "model_comparison.csv"
    if comparison_path.exists():
        print("Refitting the CoxNet baseline...")
        committed = pd.read_csv(comparison_path).set_index("model")
        coxnet_oof, _ = run_coxnet_cv(data, TrainConfig())
        current = comparison_table(coxnet_oof, oof).set_index("model")
        checks += [
            (COXNET_ROW, column, float(committed.loc[COXNET_ROW, column]), float(current.loc[COXNET_ROW, column]))
            for column in COMPARED_COLUMNS
        ]

    ok = True
    print(f"\n{'model':<44}{'metric':<10}{'committed':>11}{'fresh':>10}{'delta':>10}")
    print("-" * 85)
    for model, column, expected, actual in checks:
        delta = actual - expected
        flag = "" if abs(delta) <= args.tolerance else "  <-- DRIFT"
        if flag:
            ok = False
        print(f"{model:<44}{column:<10}{expected:>11.4f}{actual:>10.4f}{delta:>+10.4f}{flag}")

    print("-" * 85)
    print(
        f"\n{'PASS' if ok else 'FAIL'}: results reproduce within "
        f"{args.tolerance:.4f}" if ok else
        f"\nFAIL: at least one metric moved by more than {args.tolerance:.4f}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
