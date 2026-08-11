"""Evaluation: discrimination, calibration, and what the uncertainty buys us.

Everything here consumes the out-of-fold prediction table written by
:mod:`src.train`, so every number is computed on patients the model never
saw during fitting.

The time-dependent AUC and the D-calibration test are implemented directly
rather than pulled from a survival package: both are short, and it keeps the
dependency footprint small enough that the whole pipeline installs from
``requirements.txt`` without a compiler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy import stats

FIVE_YEARS_MONTHS = 60.0
DEFAULT_REFERRAL_GRID = np.round(np.arange(0.0, 0.55, 0.05), 2)


def harrell_cindex(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Concordance between predicted risk and observed survival."""
    return float(concordance_index(time, -np.asarray(risk), event))


def bootstrap_cindex_ci(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the C-index."""
    rng = np.random.default_rng(seed)
    n = len(time)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if event[idx].sum() < 2:
            continue
        values.append(harrell_cindex(risk[idx], time[idx], event[idx]))
    if not values:
        return (np.nan, np.nan)
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _censoring_survival(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of the *censoring* distribution G(t)."""
    order = np.argsort(time, kind="mergesort")
    t_sorted, e_sorted = time[order], event[order]
    uniq, first = np.unique(t_sorted, return_index=True)

    n = len(t_sorted)
    at_risk = n - first  # number with follow-up >= t
    censored = np.add.reduceat(1.0 - e_sorted, first)

    factors = np.where(at_risk > 0, 1.0 - censored / at_risk, 1.0)
    return uniq, np.cumprod(factors)


def _eval_step(times_grid: np.ndarray, values: np.ndarray, query: np.ndarray, strict: bool):
    """Evaluate a right-continuous step function; ``strict`` gives the left limit."""
    side = "left" if strict else "right"
    idx = np.searchsorted(times_grid, query, side=side) - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], 1.0)


def cumulative_dynamic_auc(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    t_eval: float = FIVE_YEARS_MONTHS,
) -> float:
    """Time-dependent AUC at ``t_eval`` with inverse-probability-of-censoring weights.

    Cases are patients who died by ``t_eval``; controls are patients known to be
    alive at ``t_eval``. Case weights 1/G(T_i-) correct for the fact that
    patients censored early are under-represented among the observed deaths.
    """
    risk, time, event = map(np.asarray, (risk, time, event))
    grid, g_values = _censoring_survival(time, event)

    cases = (time <= t_eval) & (event == 1)
    controls = time > t_eval
    if cases.sum() == 0 or controls.sum() == 0:
        return np.nan

    g_at_case = _eval_step(grid, g_values, time[cases], strict=True)
    weights = np.where(g_at_case > 0, 1.0 / g_at_case, 0.0)

    case_risk = risk[cases][:, None]
    control_risk = risk[controls][None, :]
    concordant = (case_risk > control_risk) + 0.5 * (case_risk == control_risk)

    numerator = float((weights[:, None] * concordant).sum())
    denominator = float(weights.sum() * controls.sum())
    return numerator / denominator if denominator > 0 else np.nan


def ipcw_brier_contributions(
    surv_at_t: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    t_eval: float = FIVE_YEARS_MONTHS,
) -> np.ndarray:
    """Per-patient contributions to the IPCW Brier score at ``t_eval``.

    Unlike the C-index, this is a proper scoring rule evaluated one patient at
    a time, so averaging it over a subgroup is meaningful — a subgroup's score
    does not depend on how much risk spread the subgroup happens to contain.
    That makes it the right instrument for asking whether the model really is
    less accurate where it says it is uncertain.
    """
    surv_at_t, time, event = map(np.asarray, (surv_at_t, time, event))
    grid, g_values = _censoring_survival(time, event)
    g_at_event = _eval_step(grid, g_values, time, strict=True)
    g_at_t = float(_eval_step(grid, g_values, np.array([t_eval]), strict=False)[0])

    contributions = np.zeros(len(time))
    died = (time <= t_eval) & (event == 1)
    survived = time > t_eval
    contributions[died] = surv_at_t[died] ** 2 / np.maximum(g_at_event[died], 1e-8)
    contributions[survived] = (1.0 - surv_at_t[survived]) ** 2 / max(g_at_t, 1e-8)
    return contributions


@dataclass
class DCalibrationResult:
    bin_counts: np.ndarray
    expected: float
    chi2: float
    p_value: float
    n_bins: int


def d_calibration(
    surv_at_obs_time: np.ndarray,
    event: np.ndarray,
    n_bins: int = 10,
) -> DCalibrationResult:
    """Distribution calibration test of Haider et al. (2020).

    If the predicted survival distributions are calibrated, then S(T_i | x_i)
    is uniform on [0, 1] across patients. Uncensored patients contribute their
    whole mass to one bin; a patient censored at S_i is only known to fall
    somewhere below S_i, so their mass is spread uniformly over that range.

    A *large* p-value is the good outcome: it means uniformity is not rejected.
    """
    surv = np.clip(np.asarray(surv_at_obs_time, dtype=float), 0.0, 1.0)
    event = np.asarray(event)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins)

    observed = event == 1
    if observed.any():
        idx = np.clip(np.digitize(surv[observed], edges[1:-1], right=False), 0, n_bins - 1)
        counts += np.bincount(idx, minlength=n_bins)

    for s in surv[~observed]:
        if s <= 0:
            counts[0] += 1.0
            continue
        k = min(int(np.digitize(s, edges[1:-1], right=False)), n_bins - 1)
        counts[k] += (s - edges[k]) / s
        if k > 0:
            counts[:k] += (1.0 / n_bins) / s

    expected = len(surv) / n_bins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p_value = float(stats.chi2.sf(chi2, df=n_bins - 1))
    return DCalibrationResult(counts, expected, chi2, p_value, n_bins)


@dataclass
class StratificationResult:
    groups: np.ndarray  # 0 = low risk, 1 = high risk
    logrank_p: float
    hazard_ratio: float
    hr_ci: tuple[float, float]
    median_survival: dict[str, float]


def risk_stratification(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> StratificationResult:
    """Split at the median risk score and compare the two arms."""
    risk = np.asarray(risk)
    groups = (risk > np.median(risk)).astype(int)

    low, high = groups == 0, groups == 1
    test = logrank_test(time[low], time[high], event[low], event[high])

    cph = CoxPHFitter().fit(
        pd.DataFrame({"group": groups, "_time": time, "_event": event}),
        duration_col="_time",
        event_col="_event",
    )
    hr = float(np.exp(cph.params_["group"]))
    ci = cph.confidence_intervals_.loc["group"].to_numpy()

    medians = {}
    for name, mask in (("low_risk", low), ("high_risk", high)):
        from lifelines import KaplanMeierFitter

        km = KaplanMeierFitter().fit(time[mask], event[mask])
        medians[name] = float(km.median_survival_time_)

    return StratificationResult(
        groups=groups,
        logrank_p=float(test.p_value),
        hazard_ratio=hr,
        hr_ci=(float(np.exp(ci[0])), float(np.exp(ci[1]))),
        median_survival=medians,
    )


def residual_uncertainty(
    sigma: np.ndarray,
    risk: np.ndarray,
    fold: np.ndarray,
) -> np.ndarray:
    """Strip the risk-magnitude component out of the MC-dropout spread.

    Dropout noise propagates through the network roughly in proportion to how
    far a patient's prediction sits from the cohort centre, so the raw spread
    is largely a restatement of |risk|: on METABRIC the two correlate at
    r ~ 0.4. That component says nothing about how *hard* a patient is, and it
    makes referral rules select on risk rather than on difficulty.

    Regressing sigma on a quadratic in the risk score, fold by fold, and
    keeping the residual leaves the part of the spread that is not explained by
    where the patient sits on the risk scale. No survival outcome enters this
    adjustment, so it cannot leak label information.
    """
    sigma, risk, fold = np.asarray(sigma), np.asarray(risk), np.asarray(fold)
    residual = np.full(sigma.shape, np.nan)
    for f in np.unique(fold):
        mask = fold == f
        design = np.column_stack([np.ones(mask.sum()), risk[mask], risk[mask] ** 2])
        coefficients, *_ = np.linalg.lstsq(design, sigma[mask], rcond=None)
        residual[mask] = sigma[mask] - design @ coefficients
    return residual


def _fold_groups(fold: np.ndarray | None, n: int) -> list[np.ndarray]:
    """Index groups to analyse separately; a single group when folds are unknown."""
    if fold is None:
        return [np.arange(n)]
    fold = np.asarray(fold)
    return [np.flatnonzero(fold == f) for f in np.unique(fold)]


def cindex_by_uncertainty(
    risk: np.ndarray,
    sigma: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    fold: np.ndarray | None = None,
    brier: np.ndarray | None = None,
    n_bins: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """C-index within terciles of epistemic uncertainty, binned inside each fold.

    The model should be more accurate on the patients it is confident about;
    if it is not, the uncertainty estimate is not carrying information.
    Terciles are formed fold by fold because each fold's network settles on its
    own uncertainty scale, so a single cohort-wide cut would mostly sort
    patients by which fold happened to hold them out.
    """
    risk, sigma = np.asarray(risk), np.asarray(sigma)
    labels = ["low", "medium", "high"] if n_bins == 3 else [f"q{i + 1}" for i in range(n_bins)]

    members: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
    for idx in _fold_groups(fold, len(risk)):
        ranks = pd.qcut(sigma[idx], n_bins, labels=False, duplicates="drop")
        for b in range(n_bins):
            members[b].append(idx[ranks == b])

    rows = []
    for b in range(n_bins):
        idx = np.concatenate(members[b])
        row = {
            "uncertainty_bin": labels[b],
            "n": len(idx),
            "mean_sigma": float(sigma[idx].mean()),
            "cindex": harrell_cindex(risk[idx], time[idx], event[idx]),
        }
        row["ci_low"], row["ci_high"] = bootstrap_cindex_ci(
            risk[idx], time[idx], event[idx], seed=seed
        )
        if brier is not None:
            row["brier_5y"] = float(np.asarray(brier)[idx].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def referral_curve(
    risk: np.ndarray,
    sigma: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    fold: np.ndarray | None = None,
    fractions: np.ndarray = DEFAULT_REFERRAL_GRID,
    n_random: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Accuracy as a function of how many uncertain patients are deferred.

    At each referral fraction the most uncertain patients are handed to a
    clinician and the C-index is recomputed on the retained ones. Referral is
    applied within each fold and the resulting C-indices averaged, so a fold
    with a wider uncertainty scale cannot dominate the deferred set.

    The random referral arm is the control: it shows how much of any movement
    is simply an artefact of evaluating on a smaller cohort.
    """
    risk, sigma, time, event = map(np.asarray, (risk, sigma, time, event))
    rng = np.random.default_rng(seed)
    groups = _fold_groups(fold, len(risk))

    rows = []
    for frac in fractions:
        per_fold_unc, retained = [], 0
        # Each random draw is scored the same way the real rule is - deferred
        # within every fold, then averaged - so the spread below describes the
        # control statistic itself rather than fold-to-fold variation.
        random_repeats = np.zeros(n_random)
        usable = True

        for idx in groups:
            n = len(idx)
            keep = n - int(round(frac * n))
            if keep < 20:
                usable = False
                break
            retained += keep

            kept = idx[np.argsort(sigma[idx])[:keep]]
            per_fold_unc.append(harrell_cindex(risk[kept], time[kept], event[kept]))

            for r in range(n_random):
                ridx = idx[rng.choice(n, size=keep, replace=False)]
                random_repeats[r] += harrell_cindex(risk[ridx], time[ridx], event[ridx])

        if not usable:
            continue
        random_repeats /= len(groups)
        rows.append(
            {
                "referral_fraction": float(frac),
                "n_retained": retained,
                "cindex_uncertainty_referral": float(np.mean(per_fold_unc)),
                "cindex_random_referral": float(random_repeats.mean()),
                "cindex_random_sd": float(random_repeats.std()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(oof: pd.DataFrame, referral_fraction: float = 0.2, seed: int = 0) -> dict:
    """Assemble every headline number from the out-of-fold prediction table."""
    time = oof["time"].to_numpy()
    event = oof["event"].to_numpy()

    # Fold-standardised scores: risk is only identified up to scale and shift
    # within a fold, so pooling the raw scores across folds would compare
    # patients on incompatible scales.
    risk = oof["risk_z"].to_numpy()
    sigma = oof["sigma_z"].to_numpy()
    baseline = oof["baseline_risk_z"].to_numpy()

    results: dict = {}

    results["baseline"] = {
        "cindex": harrell_cindex(baseline, time, event),
        "cindex_ci": bootstrap_cindex_ci(baseline, time, event, seed=seed),
        "auc_5y": cumulative_dynamic_auc(baseline, time, event),
        "dcal": d_calibration(oof["baseline_surv_at_obs_time"].to_numpy(), event),
    }
    results["bayesian"] = {
        "cindex": harrell_cindex(risk, time, event),
        "cindex_ci": bootstrap_cindex_ci(risk, time, event, seed=seed),
        "auc_5y": cumulative_dynamic_auc(risk, time, event),
        "dcal": d_calibration(oof["surv_at_obs_time"].to_numpy(), event),
    }

    fold = oof["fold"].to_numpy()
    sigma_resid = residual_uncertainty(sigma, risk, fold)
    brier = ipcw_brier_contributions(oof["surv_at_5y"].to_numpy(), time, event)

    results["uncertainty_bins"] = cindex_by_uncertainty(
        risk, sigma_resid, time, event, fold=fold, brier=brier, seed=seed
    )
    results["uncertainty_bins_raw"] = cindex_by_uncertainty(
        risk, sigma, time, event, fold=fold, brier=brier, seed=seed
    )
    results["referral_curve"] = referral_curve(risk, sigma_resid, time, event, fold=fold, seed=seed)
    results["referral_curve_raw"] = referral_curve(risk, sigma, time, event, fold=fold, seed=seed)

    # Referral arm: defer the least-certain patients inside each fold, then
    # average, matching how the referral curve is built.
    curve = results["referral_curve"].set_index("referral_fraction")
    reference_c = float(curve.loc[0.0, "cindex_uncertainty_referral"])
    at_fraction = curve.loc[referral_fraction]

    retained = np.concatenate(
        [
            idx[np.argsort(sigma_resid[idx])[: len(idx) - int(round(referral_fraction * len(idx)))]]
            for idx in _fold_groups(fold, len(risk))
        ]
    )
    results["referral"] = {
        "fraction": referral_fraction,
        "n_retained": int(at_fraction["n_retained"]),
        "cindex": float(at_fraction["cindex_uncertainty_referral"]),
        "cindex_ci": bootstrap_cindex_ci(
            risk[retained], time[retained], event[retained], seed=seed
        ),
        "auc_5y": cumulative_dynamic_auc(risk[retained], time[retained], event[retained]),
        "reference_cindex": reference_c,
        "gain": float(at_fraction["cindex_uncertainty_referral"]) - reference_c,
    }

    results["stratification"] = risk_stratification(risk, time, event)
    results["correlation_sigma_risk"] = float(np.corrcoef(sigma, np.abs(risk))[0, 1])
    results["sigma_summary"] = {
        "mean": float(sigma.mean()),
        "median": float(np.median(sigma)),
        "iqr": (float(np.quantile(sigma, 0.25)), float(np.quantile(sigma, 0.75))),
    }
    return results


def performance_table(results: dict) -> pd.DataFrame:
    """The headline table, one row per model."""
    rows = []
    for key, label in (
        ("baseline", "Clinical Cox baseline"),
        ("bayesian", "Bayesian MLP (MC Dropout)"),
        ("referral", "+ Uncertainty referral (top 20% deferred)"),
    ):
        r = results[key]
        dcal = r.get("dcal")
        rows.append(
            {
                "model": label,
                "n": r.get("n_retained", ""),
                "cindex": round(r["cindex"], 4),
                "cindex_ci_low": round(r["cindex_ci"][0], 4),
                "cindex_ci_high": round(r["cindex_ci"][1], 4),
                "auc_5y": round(r["auc_5y"], 4),
                "dcal_p_value": round(dcal.p_value, 4) if dcal is not None else "",
                "referral_gain": round(r["gain"], 4) if "gain" in r else "",
            }
        )
    return pd.DataFrame(rows)
