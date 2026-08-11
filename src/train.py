"""Cross-validated training of the Bayesian survival model.

Runs stratified k-fold CV over the cohort. Within every fold the gene
selection, scaling, network and Cox baseline are all fitted on the training
patients only, and the held-out patients contribute one row each to an
out-of-fold prediction table. Evaluating that table gives a single
cohort-level estimate of performance rather than a mean over folds.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from .data_loader import FoldPreprocessor, METABRICData, load_metabric
from .model import (
    BayesianMLP,
    BreslowBaseline,
    cox_partial_likelihood_loss,
    predict_deterministic,
    predict_mc,
)

FIVE_YEARS_MONTHS = 60.0


@dataclass
class TrainConfig:
    """Hyperparameters for the full cross-validated run."""

    # Selected by grid search on the inner validation C-index; see README.
    n_genes: int = 1000
    hidden_dims: tuple[int, ...] = (256, 64)
    dropout: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 800
    patience: int = 100
    n_folds: int = 5
    n_mc_samples: int = 200
    val_fraction: float = 0.15
    calibration_fraction: float = 0.15
    clinical_shortcut: bool = True
    baseline_penalizer: float = 0.1
    seed: int = 42
    device: str = "cpu"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    val_cindex: float
    test_cindex: float
    baseline_cindex: float
    history: list[dict] = field(default_factory=list)


def configure_logging(log_file: Path, verbose: bool = True) -> None:
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )


def log_environment(cfg: TrainConfig, data: METABRICData) -> None:
    logger.info("=" * 72)
    logger.info("METABRIC Bayesian survival model — cross-validated training run")
    logger.info("=" * 72)
    logger.info(f"python      : {platform.python_version()} ({platform.platform()})")
    logger.info(f"torch       : {torch.__version__}  |  numpy: {np.__version__}")
    logger.info(f"threads     : {torch.get_num_threads()}")
    logger.info(
        f"cohort      : {data.n_samples} patients, {data.expression.shape[1]} genes, "
        f"{data.clinical.shape[1]} clinical covariates"
    )
    logger.info(
        f"events      : {int(data.event.sum())} deaths "
        f"({100 * data.event.mean():.1f}%), median follow-up "
        f"{np.median(data.time):.1f} months"
    )
    logger.info("hyperparameters:")
    for key, value in cfg.as_dict().items():
        logger.info(f"    {key:<20} = {value}")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _cindex(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index. Risk is a log hazard ratio, so higher means worse."""
    return float(concordance_index(time, -risk, event))


def train_fold(
    X_train: np.ndarray,
    y_train: tuple[np.ndarray, np.ndarray],
    X_val: np.ndarray,
    y_val: tuple[np.ndarray, np.ndarray],
    cfg: TrainConfig,
    fold: int,
    checkpoint_path: Path | None = None,
    n_clinical: int = 0,
) -> tuple[BayesianMLP, int, float, list[dict]]:
    """Train one fold, early-stopping on validation concordance."""
    device = torch.device(cfg.device)
    model = BayesianMLP(
        input_dim=X_train.shape[1],
        hidden_dims=tuple(cfg.hidden_dims),
        dropout=cfg.dropout,
        n_clinical=n_clinical if cfg.clinical_shortcut else 0,
    ).to(device)

    # The clinical shortcut is a handful of well-supported coefficients; the
    # shrinkage that keeps the wide deep branch in check would only bias it.
    shortcut_params = (
        list(model.clinical_head.parameters()) if model.clinical_head is not None else []
    )
    shortcut_ids = {id(p) for p in shortcut_params}
    deep_params = [p for p in model.parameters() if id(p) not in shortcut_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": deep_params, "weight_decay": cfg.weight_decay},
            {"params": shortcut_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )

    t_train, e_train = y_train
    t_val, e_val = y_val

    # The partial likelihood couples every patient through the risk set, so the
    # whole training fold is a single batch.
    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    tt = torch.as_tensor(t_train, dtype=torch.float32, device=device)
    et = torch.as_tensor(e_train, dtype=torch.float32, device=device)

    best_val, best_epoch, best_state = -np.inf, 0, None
    history: list[dict] = []
    epochs_without_gain = 0

    progress = tqdm(
        range(1, cfg.max_epochs + 1),
        desc=f"fold {fold}",
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),  # keep redirected logs readable
    )
    for epoch in progress:
        model.train()
        optimizer.zero_grad()
        loss = cox_partial_likelihood_loss(model(Xt), tt, et)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_risk = predict_deterministic(model, X_val, device=device)
        val_cindex = _cindex(val_risk, t_val, e_val)
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_cindex": val_cindex})

        if val_cindex > best_val + 1e-5:
            best_val, best_epoch = val_cindex, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1

        progress.set_postfix(loss=f"{loss.item():.4f}", val_c=f"{val_cindex:.4f}")
        if epoch % 50 == 0 or epoch == 1:
            logger.debug(
                f"fold {fold} | epoch {epoch:4d} | loss {loss.item():.4f} | "
                f"val C-index {val_cindex:.4f}"
            )
        if epochs_without_gain >= cfg.patience:
            logger.debug(f"fold {fold} | early stop at epoch {epoch} (patience {cfg.patience})")
            break
    progress.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fold": fold,
                "state_dict": model.state_dict(),
                "input_dim": X_train.shape[1],
                "config": cfg.as_dict(),
                "best_epoch": best_epoch,
                "val_cindex": best_val,
            },
            checkpoint_path,
        )
    return model, best_epoch, best_val, history


def fit_calibration_slope(
    risk_calib: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """Van Houwelingen's calibration slope for a risk score.

    A model fitted on 1.3k patients with 1000 gene features produces risk
    scores that are more spread out than the data support, which shows up as
    survival probabilities piled at both ends of the scale. Regressing the
    held-out outcomes on the score alone recovers the shrinkage factor: a
    slope below 1 means the predictions are over-dispersed.

    The map is monotone, so discrimination is untouched — only the absolute
    probabilities move.
    """
    centred = np.asarray(risk_calib, dtype=np.float64)
    centred = centred - centred.mean()
    if np.std(centred) < 1e-8 or event.sum() < 5:
        return 1.0
    fitted = CoxPHFitter().fit(
        pd.DataFrame({"risk": centred, "_time": time, "_event": event}),
        duration_col="_time",
        event_col="_event",
    )
    slope = float(fitted.params_["risk"])
    return float(np.clip(slope, 0.05, 2.0))


def fit_clinical_baseline(
    clinical_train: pd.DataFrame,
    t_train: np.ndarray,
    e_train: np.ndarray,
    clinical_test: pd.DataFrame,
    penalizer: float,
    clinical_reference: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Penalised Cox model on clinical covariates only.

    Returns the log hazard ratio for the reference patients (those that
    calibrate the baseline hazard, defaulting to the training patients) and
    for the test patients.
    """
    df = clinical_train.copy()
    df["_time"], df["_event"] = t_train, e_train
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="_time", event_col="_event")
    reference = clinical_train if clinical_reference is None else clinical_reference
    reference_risk = np.log(cph.predict_partial_hazard(reference).to_numpy())
    test_risk = np.log(cph.predict_partial_hazard(clinical_test).to_numpy())
    return reference_risk, test_risk


def run_cross_validation(
    data: METABRICData,
    cfg: TrainConfig,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Full stratified k-fold run, returning the out-of-fold prediction table."""
    set_seed(cfg.seed)
    splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    columns = [
        "risk_mean",
        "risk_std",
        "risk_z",
        "sigma_z",
        "risk_deterministic",
        "baseline_risk",
        "baseline_risk_z",
        "surv_at_obs_time",
        "surv_at_5y",
        "baseline_surv_at_obs_time",
        "baseline_surv_at_5y",
    ]
    oof = pd.DataFrame(
        np.full((data.n_samples, len(columns)), np.nan),
        columns=columns,
        index=data.sample_ids,
    )
    oof["fold"] = -1

    fold_results: list[FoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(data.n_samples), data.event), start=1
    ):
        started = time.time()

        # Split the training fold three ways. The early-stopping slice cannot
        # double as the calibration slice: the epoch is *chosen* to look good
        # on it, which leaves its risk scores mildly optimistic and drags the
        # baseline hazard down with them. The calibration slice is touched by
        # nothing except the Breslow fit.
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

        pre = FoldPreprocessor(n_genes=cfg.n_genes).fit(data, inner_train)
        X_tr = pre.transform(data, inner_train)
        X_va = pre.transform(data, inner_val)
        X_ca = pre.transform(data, inner_calib)
        X_te = pre.transform(data, test_idx)

        logger.info(
            f"[fold {fold}/{cfg.n_folds}] train={len(inner_train)} val={len(inner_val)} "
            f"calib={len(inner_calib)} test={len(test_idx)} | features={X_tr.shape[1]}"
        )

        ckpt = None if checkpoint_dir is None else checkpoint_dir / f"fold_{fold}.pt"
        model, best_epoch, best_val, history = train_fold(
            X_tr,
            (data.time[inner_train], data.event[inner_train]),
            X_va,
            (data.time[inner_val], data.event[inner_val]),
            cfg,
            fold,
            checkpoint_path=ckpt,
            n_clinical=pre.n_clinical_,
        )

        draws = predict_mc(
            model, X_te, n_samples=cfg.n_mc_samples, device=cfg.device, seed=cfg.seed + fold
        )
        risk_mean = draws.mean(axis=0)
        risk_std = draws.std(axis=0, ddof=1)
        risk_det = predict_deterministic(model, X_te, device=cfg.device)

        # Absolute survival probabilities are calibrated on a split the network
        # never saw and that played no part in choosing the epoch. Two things
        # are estimated there: a shrinkage slope for the over-dispersed risk
        # score, and the baseline hazard itself. The Cox baseline goes through
        # exactly the same procedure, so the two are compared on equal terms.
        calib_draws = predict_mc(
            model,
            X_ca,
            n_samples=cfg.n_mc_samples,
            device=cfg.device,
            seed=cfg.seed + fold,
        )
        calib_risk_mean = calib_draws.mean(axis=0)
        slope = fit_calibration_slope(
            calib_risk_mean, data.time[inner_calib], data.event[inner_calib]
        )
        calib_centre = calib_risk_mean.mean()
        breslow = BreslowBaseline().fit(
            slope * (calib_risk_mean - calib_centre),
            data.time[inner_calib],
            data.event[inner_calib],
        )
        risk_calibrated = slope * (risk_mean - calib_centre)

        baseline_calib_risk, baseline_risk = fit_clinical_baseline(
            pre.transform_clinical_only(data, inner_train),
            data.time[inner_train],
            data.event[inner_train],
            pre.transform_clinical_only(data, test_idx),
            cfg.baseline_penalizer,
            clinical_reference=pre.transform_clinical_only(data, inner_calib),
        )
        base_slope = fit_calibration_slope(
            baseline_calib_risk, data.time[inner_calib], data.event[inner_calib]
        )
        base_calib_centre = baseline_calib_risk.mean()
        baseline_breslow = BreslowBaseline().fit(
            base_slope * (baseline_calib_risk - base_calib_centre),
            data.time[inner_calib],
            data.event[inner_calib],
        )
        baseline_calibrated = base_slope * (baseline_risk - base_calib_centre)
        logger.debug(
            f"fold {fold} | calibration slope: MLP {slope:.3f}, clinical Cox {base_slope:.3f}"
        )

        # A Cox risk score is only identified up to an additive constant, and
        # each fold's network settles on its own scale, so the raw scores from
        # different folds cannot be pooled into one ranking. Referring every
        # fold to its own training distribution puts them on a common footing;
        # this is a monotone map, so within-fold concordance is untouched.
        train_centre, train_scale = calib_risk_mean.mean(), calib_risk_mean.std()
        base_centre, base_scale = baseline_calib_risk.mean(), baseline_calib_risk.std()

        ids = data.sample_ids[test_idx]
        oof.loc[ids, "risk_mean"] = risk_mean
        oof.loc[ids, "risk_std"] = risk_std
        oof.loc[ids, "risk_z"] = (risk_mean - train_centre) / train_scale
        oof.loc[ids, "sigma_z"] = risk_std / train_scale
        oof.loc[ids, "risk_deterministic"] = risk_det
        oof.loc[ids, "baseline_risk"] = baseline_risk
        oof.loc[ids, "baseline_risk_z"] = (baseline_risk - base_centre) / base_scale
        oof.loc[ids, "surv_at_obs_time"] = breslow.survival_at(
            risk_calibrated, data.time[test_idx]
        )
        oof.loc[ids, "surv_at_5y"] = breslow.survival_function(
            risk_calibrated, np.array([FIVE_YEARS_MONTHS])
        ).ravel()
        oof.loc[ids, "baseline_surv_at_obs_time"] = baseline_breslow.survival_at(
            baseline_calibrated, data.time[test_idx]
        )
        oof.loc[ids, "baseline_surv_at_5y"] = baseline_breslow.survival_function(
            baseline_calibrated, np.array([FIVE_YEARS_MONTHS])
        ).ravel()
        oof.loc[ids, "fold"] = fold

        test_c = _cindex(risk_mean, data.time[test_idx], data.event[test_idx])
        base_c = _cindex(baseline_risk, data.time[test_idx], data.event[test_idx])
        logger.info(
            f"[fold {fold}/{cfg.n_folds}] best epoch {best_epoch} | val C {best_val:.4f} | "
            f"test C {test_c:.4f} | clinical Cox C {base_c:.4f} | "
            f"mean sigma {risk_std.mean():.3f} | {time.time() - started:.1f}s"
        )
        fold_results.append(
            FoldResult(fold, best_epoch, best_val, test_c, base_c, history)
        )

    oof.insert(0, "time", data.time)
    oof.insert(1, "event", data.event)
    oof.index.name = "sample_id"

    mean_c = np.mean([f.test_cindex for f in fold_results])
    std_c = np.std([f.test_cindex for f in fold_results])
    pooled_c = _cindex(oof["risk_z"].to_numpy(), data.time, data.event)
    pooled_base = _cindex(oof["baseline_risk_z"].to_numpy(), data.time, data.event)
    logger.info("-" * 72)
    logger.info(f"per-fold C-index    : {mean_c:.4f} +/- {std_c:.4f}")
    logger.info(f"pooled OOF C-index  : {pooled_c:.4f} (Bayesian MLP)")
    logger.info(f"pooled OOF C-index  : {pooled_base:.4f} (clinical Cox baseline)")
    logger.info("-" * 72)
    return oof, fold_results


def fit_final_model(
    data: METABRICData,
    cfg: TrainConfig,
    checkpoint_path: Path | None = None,
) -> tuple[BayesianMLP, FoldPreprocessor, np.ndarray]:
    """Refit a single model on the whole cohort.

    Cross-validation gives the performance estimate; this model exists only so
    that the attribution analysis describes one consistent set of weights
    rather than an average over five differently-selected gene panels.
    """
    set_seed(cfg.seed)
    all_idx = np.arange(data.n_samples)
    inner_train, inner_val = train_test_split(
        all_idx,
        test_size=cfg.val_fraction,
        random_state=cfg.seed,
        stratify=data.event,
    )

    pre = FoldPreprocessor(n_genes=cfg.n_genes).fit(data, inner_train)
    X_tr = pre.transform(data, inner_train)
    X_va = pre.transform(data, inner_val)

    logger.info(f"[final] refitting on all {data.n_samples} patients for attribution")
    model, best_epoch, best_val, _ = train_fold(
        X_tr,
        (data.time[inner_train], data.event[inner_train]),
        X_va,
        (data.time[inner_val], data.event[inner_val]),
        cfg,
        fold=0,
        checkpoint_path=checkpoint_path,
        n_clinical=pre.n_clinical_,
    )
    logger.info(f"[final] best epoch {best_epoch} | val C-index {best_val:.4f}")
    return model, pre, pre.transform(data, all_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Bayesian survival model on METABRIC.")
    parser.add_argument("--data-dir", default="data/brca_metabric")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-file", default="logs/training.log")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--n-genes", type=int, default=TrainConfig.n_genes)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--n-folds", type=int, default=TrainConfig.n_folds)
    parser.add_argument("--mc-samples", type=int, default=TrainConfig.n_mc_samples)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        n_genes=args.n_genes,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        n_folds=args.n_folds,
        n_mc_samples=args.mc_samples,
        seed=args.seed,
    )

    configure_logging(Path(args.log_file), verbose=not args.quiet)
    data = load_metabric(args.data_dir)
    log_environment(cfg, data)

    oof, folds = run_cross_validation(data, cfg, checkpoint_dir=Path(args.checkpoint_dir))

    results_dir = Path(args.results_dir)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    oof.to_csv(results_dir / "oof_predictions.csv")

    fold_table = pd.DataFrame(
        [
            {
                "fold": f.fold,
                "best_epoch": f.best_epoch,
                "val_cindex": f.val_cindex,
                "test_cindex": f.test_cindex,
                "baseline_cindex": f.baseline_cindex,
            }
            for f in folds
        ]
    )
    fold_table.to_csv(results_dir / "tables" / "fold_metrics.csv", index=False)
    (results_dir / "run_config.json").write_text(json.dumps(cfg.as_dict(), indent=2))
    logger.info(f"out-of-fold predictions written to {results_dir / 'oof_predictions.csv'}")


if __name__ == "__main__":
    main()
