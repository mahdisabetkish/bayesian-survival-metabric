"""Elastic-net Cox (CoxNet) on exactly the features the network sees.

The main run compares two models that differ in two ways at once: a penalised
Cox model on 12 clinical covariates, and an MLP on 1,000 genes plus the same
clinical block. This baseline separates the two differences. It gets the
network's inputs, folds, training patients, model-selection split and
calibration procedure, but it is linear in the features. If it matches the
network, the network's extra capacity is not what the transcriptome needs on
this cohort.

Clinical covariates are left unpenalised, mirroring the network's unshrunk
clinical shortcut; only the gene coefficients are shrunk. The elastic-net
mixing and penalty strength are chosen on the inner validation split, the same
split the network uses for early stopping.

    python -m src.coxnet_baseline
"""

from __future__ import annotations

import argparse
import time as walltime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis

from .data_loader import ENDPOINTS, FoldPreprocessor, METABRICData, load_metabric
from .model import BreslowBaseline
from .train import (
    FIVE_YEARS_MONTHS,
    TrainConfig,
    configure_logging,
    endpoint_dir,
    fit_calibration_slope,
)
from .uncertainty import (
    bootstrap_cindex_ci,
    cumulative_dynamic_auc,
    d_calibration,
    harrell_cindex,
)

L1_RATIOS = (0.1, 0.5, 0.9)
N_ALPHAS = 30
# The path stops well short of an unpenalised fit: with ~1,000 features and
# ~1,400 training patients the low end of the path barely converges and never
# wins on validation.
ALPHA_MIN_RATIO = 0.01


@dataclass
class CoxnetFoldResult:
    fold: int
    l1_ratio: float
    alpha: float
    n_genes_selected: int
    val_cindex: float
    test_cindex: float
    calibration_slope: float


def cv_splits(data: METABRICData, cfg: TrainConfig):
    """Yield the same fold and inner splits as :func:`src.train.run_cross_validation`."""
    splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(data.n_samples), data.event), start=1
    ):
        inner_train, held_out = train_test_split(
            train_idx,
            test_size=cfg.val_fraction + cfg.calibration_fraction,
            random_state=cfg.seed + fold,
            stratify=data.event[train_idx],
        )
        inner_val, inner_calib = train_test_split(
            held_out,
            test_size=cfg.calibration_fraction / (cfg.val_fraction + cfg.calibration_fraction),
            random_state=cfg.seed + fold,
            stratify=data.event[held_out],
        )
        yield fold, inner_train, inner_val, inner_calib, test_idx


def _structured(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    return np.array(
        list(zip(event.astype(bool), time)), dtype=[("event", "?"), ("time", "<f8")]
    )


def select_coxnet(
    X_train: np.ndarray,
    y_train: tuple[np.ndarray, np.ndarray],
    X_val: np.ndarray,
    y_val: tuple[np.ndarray, np.ndarray],
    n_clinical: int,
    l1_ratios: tuple[float, ...] = L1_RATIOS,
) -> tuple[CoxnetSurvivalAnalysis, float, float, float]:
    """Fit one regularisation path per mixing ratio; keep the best point on validation.

    Returns the fitted path, the chosen ``l1_ratio`` and ``alpha``, and the
    validation C-index at that point.
    """
    penalty = np.ones(X_train.shape[1])
    if n_clinical:
        penalty[-n_clinical:] = 0.0

    best = None
    for l1_ratio in l1_ratios:
        model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            n_alphas=N_ALPHAS,
            alpha_min_ratio=ALPHA_MIN_RATIO,
            penalty_factor=penalty,
            max_iter=100_000,
        )
        model.fit(X_train, _structured(*y_train))
        for alpha in model.alphas_:
            c = harrell_cindex(model.predict(X_val, alpha=alpha), *y_val)
            if best is None or c > best[3] + 1e-5:
                best = (model, l1_ratio, float(alpha), c)
    return best


def run_coxnet_cv(
    data: METABRICData,
    cfg: TrainConfig,
    l1_ratios: tuple[float, ...] = L1_RATIOS,
) -> tuple[pd.DataFrame, list[CoxnetFoldResult]]:
    """Cross-validated CoxNet, returning an out-of-fold table like the network's."""
    columns = ["risk", "risk_z", "surv_at_obs_time", "surv_at_5y"]
    oof = pd.DataFrame(
        np.full((data.n_samples, len(columns)), np.nan), columns=columns, index=data.sample_ids
    )
    oof["fold"] = -1

    results: list[CoxnetFoldResult] = []
    for fold, inner_train, inner_val, inner_calib, test_idx in cv_splits(data, cfg):
        started = walltime.time()
        pre = FoldPreprocessor(n_genes=cfg.n_genes).fit(data, inner_train)
        X_tr, X_va = pre.transform(data, inner_train), pre.transform(data, inner_val)
        X_ca, X_te = pre.transform(data, inner_calib), pre.transform(data, test_idx)

        model, l1_ratio, alpha, val_c = select_coxnet(
            X_tr,
            (data.time[inner_train], data.event[inner_train]),
            X_va,
            (data.time[inner_val], data.event[inner_val]),
            n_clinical=pre.n_clinical_,
            l1_ratios=l1_ratios,
        )
        test_risk = model.predict(X_te, alpha=alpha)
        calib_risk = model.predict(X_ca, alpha=alpha)

        # Same calibration route as the network and the clinical Cox model:
        # shrinkage slope and Breslow baseline, both on the calibration split.
        slope = fit_calibration_slope(calib_risk, data.time[inner_calib], data.event[inner_calib])
        centre = calib_risk.mean()
        breslow = BreslowBaseline().fit(
            slope * (calib_risk - centre), data.time[inner_calib], data.event[inner_calib]
        )
        calibrated = slope * (test_risk - centre)

        ids = data.sample_ids[test_idx]
        oof.loc[ids, "risk"] = test_risk
        oof.loc[ids, "risk_z"] = (test_risk - centre) / max(calib_risk.std(), 1e-8)
        oof.loc[ids, "surv_at_obs_time"] = breslow.survival_at(calibrated, data.time[test_idx])
        oof.loc[ids, "surv_at_5y"] = breslow.survival_function(
            calibrated, np.array([FIVE_YEARS_MONTHS])
        ).ravel()
        oof.loc[ids, "fold"] = fold

        alpha_idx = int(np.argmin(np.abs(model.alphas_ - alpha)))
        gene_coef = model.coef_[: len(pre.selected_genes_), alpha_idx]
        test_c = harrell_cindex(test_risk, data.time[test_idx], data.event[test_idx])
        results.append(
            CoxnetFoldResult(
                fold=fold,
                l1_ratio=l1_ratio,
                alpha=alpha,
                n_genes_selected=int(np.count_nonzero(gene_coef)),
                val_cindex=val_c,
                test_cindex=test_c,
                calibration_slope=slope,
            )
        )
        logger.info(
            f"[fold {fold}/{cfg.n_folds}] l1_ratio {l1_ratio} | alpha {alpha:.4g} | "
            f"{results[-1].n_genes_selected} genes kept | val C {val_c:.4f} | "
            f"test C {test_c:.4f} | slope {slope:.2f} | {walltime.time() - started:.1f}s"
        )

    oof.insert(0, "time", data.time)
    oof.insert(1, "event", data.event)
    oof.index.name = "sample_id"
    return oof, results


def _summarise(
    label: str,
    risk: np.ndarray,
    surv_at_obs_time: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    fold: np.ndarray,
    seed: int,
) -> dict:
    per_fold = [
        harrell_cindex(risk[fold == f], time[fold == f], event[fold == f]) for f in np.unique(fold)
    ]
    lo, hi = bootstrap_cindex_ci(risk, time, event, seed=seed)
    return {
        "model": label,
        "cindex": round(harrell_cindex(risk, time, event), 4),
        "cindex_ci_low": round(lo, 4),
        "cindex_ci_high": round(hi, 4),
        "cindex_fold_mean": round(float(np.mean(per_fold)), 4),
        "cindex_fold_sd": round(float(np.std(per_fold)), 4),
        "auc_5y": round(cumulative_dynamic_auc(risk, time, event), 4),
        "dcal_p_value": round(d_calibration(surv_at_obs_time, event).p_value, 4),
    }


def paired_bootstrap_difference(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """C-index of A minus C-index of B, with a percentile interval from paired resamples."""
    rng = np.random.default_rng(seed)
    n = len(time)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if event[idx].sum() < 2:
            continue
        diffs.append(
            harrell_cindex(risk_a[idx], time[idx], event[idx])
            - harrell_cindex(risk_b[idx], time[idx], event[idx])
        )
    point = harrell_cindex(risk_a, time, event) - harrell_cindex(risk_b, time, event)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def comparison_table(coxnet_oof: pd.DataFrame, main_oof: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Clinical Cox, CoxNet and the network side by side on the same patients."""
    if not (coxnet_oof["fold"].to_numpy() == main_oof.loc[coxnet_oof.index, "fold"].to_numpy()).all():
        raise ValueError("CoxNet folds differ from the main run's folds; the comparison is not paired.")
    main_oof = main_oof.loc[coxnet_oof.index]
    time, event = coxnet_oof["time"].to_numpy(), coxnet_oof["event"].to_numpy()
    fold = coxnet_oof["fold"].to_numpy()

    rows = [
        _summarise(
            "Clinical Cox baseline (12 covariates)",
            main_oof["baseline_risk_z"].to_numpy(),
            main_oof["baseline_surv_at_obs_time"].to_numpy(),
            time, event, fold, seed,
        ),
        _summarise(
            "CoxNet, 1,000 genes + clinical",
            coxnet_oof["risk_z"].to_numpy(),
            coxnet_oof["surv_at_obs_time"].to_numpy(),
            time, event, fold, seed,
        ),
        _summarise(
            "Bayesian MLP, 1,000 genes + clinical",
            main_oof["risk_z"].to_numpy(),
            main_oof["surv_at_obs_time"].to_numpy(),
            time, event, fold, seed,
        ),
    ]
    table = pd.DataFrame(rows)

    coxnet = coxnet_oof["risk_z"].to_numpy()
    for column, other in (
        ("coxnet_minus_clinical_cox", main_oof["baseline_risk_z"].to_numpy()),
        ("coxnet_minus_mlp", main_oof["risk_z"].to_numpy()),
    ):
        diff, lo, hi = paired_bootstrap_difference(coxnet, other, time, event, seed=seed)
        table[column] = ""
        table.loc[1, column] = f"{diff:+.4f} ({lo:+.4f} to {hi:+.4f})"
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validated CoxNet baseline on METABRIC.")
    parser.add_argument("--data-dir", default="data/brca_metabric")
    parser.add_argument(
        "--endpoint", choices=sorted(ENDPOINTS), default="os",
        help="os = overall survival (default), rfs = relapse-free survival",
    )
    parser.add_argument("--results-dir", help="default: results, or results/<endpoint>")
    parser.add_argument("--log-file", help="default: logs/coxnet.log, or logs/<endpoint>/")
    parser.add_argument("--n-genes", type=int, default=TrainConfig.n_genes)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(n_genes=args.n_genes, seed=args.seed)
    log_file = args.log_file or endpoint_dir("logs", args.endpoint) / "coxnet.log"
    configure_logging(Path(log_file), verbose=not args.quiet)
    data = load_metabric(args.data_dir, endpoint=args.endpoint)
    logger.info(
        f"CoxNet baseline | endpoint {args.endpoint} | {data.n_samples} patients | "
        f"{cfg.n_genes} genes per fold + {data.clinical.shape[1]} clinical (unpenalised) | "
        f"l1_ratio grid {L1_RATIOS}"
    )

    oof, folds = run_coxnet_cv(data, cfg)

    results_dir = Path(args.results_dir or endpoint_dir("results", args.endpoint))
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(results_dir / "coxnet_oof_predictions.csv")
    pd.DataFrame([f.__dict__ for f in folds]).to_csv(
        tables_dir / "coxnet_fold_metrics.csv", index=False
    )

    main_oof_path = results_dir / "oof_predictions.csv"
    if not main_oof_path.exists():
        logger.warning(
            f"{main_oof_path} not found; run `python -m src.train --endpoint {args.endpoint}` "
            "for the comparison table."
        )
        return
    table = comparison_table(oof, pd.read_csv(main_oof_path, index_col="sample_id"))
    table.to_csv(tables_dir / "model_comparison.csv", index=False)
    logger.info("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
