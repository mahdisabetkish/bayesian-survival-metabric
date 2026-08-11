"""Publication figures and result tables.

Reads the out-of-fold prediction table produced by :mod:`src.train`, runs the
evaluation in :mod:`src.uncertainty`, and writes every figure and CSV that the
README reports. Run as ``python -m src.visualize``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from loguru import logger

from .data_loader import load_metabric
from .train import TrainConfig, configure_logging, fit_final_model
from .uncertainty import FIVE_YEARS_MONTHS, evaluate, performance_table

# Two-class categorical pair (seaborn "muted"); checked for colour-vision
# deficiency separation before use. Uncertainty is an ordered quantity, so its
# bars use a single-hue sequential ramp instead.
COLOR_LOW = "#4878d0"
COLOR_HIGH = "#d65f5f"
COLOR_RAW = "#c1651b"
COLOR_NEUTRAL = "#8c8c8c"  # reserved for the control arm, never for a real series
SEQUENTIAL_BLUES = ["#c6d7f0", "#7ba0dc", "#2f5597"]
SEQUENTIAL_REDS = ["#f4cbc4", "#e08f81", "#b5442f"]

DPI = 300


def set_style() -> None:
    sns.set_theme(style="whitegrid", palette="muted", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#bdbdbd",
            "grid.color": "#e5e5e5",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def _format_p(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def plot_km_curves(oof: pd.DataFrame, results: dict, out_path: Path) -> None:
    """Kaplan-Meier survival by model-predicted risk group."""
    strat = results["stratification"]
    time, event = oof["time"].to_numpy(), oof["event"].to_numpy()
    groups = strat.groups

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    fitters = []
    for label, mask, color in (
        ("Low risk (below median)", groups == 0, COLOR_LOW),
        ("High risk (above median)", groups == 1, COLOR_HIGH),
    ):
        km = KaplanMeierFitter(label=label)
        km.fit(time[mask], event[mask])
        km.plot_survival_function(ax=ax, color=color, linewidth=2.0, ci_alpha=0.15)
        fitters.append(km)

    ax.set_xlabel("Months since diagnosis")
    ax.set_ylabel("Overall survival probability")
    ax.set_title("Risk stratification by the Bayesian model")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 300)

    annotation = (
        f"log-rank {_format_p(strat.logrank_p)}\n"
        f"HR = {strat.hazard_ratio:.2f} "
        f"({strat.hr_ci[0]:.2f}–{strat.hr_ci[1]:.2f})\n"
        f"median OS: {strat.median_survival['low_risk']:.0f} vs "
        f"{strat.median_survival['high_risk']:.0f} months"
    )
    ax.text(
        0.03,
        0.06,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#d0d0d0"),
    )
    ax.legend(loc="upper right")
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def plot_calibration(oof: pd.DataFrame, results: dict, out_path: Path) -> None:
    """Two views of calibration: the D-calibration histogram and a 5-year reliability plot."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- (a) D-calibration ------------------------------------------------
    dcal = results["bayesian"]["dcal"]
    ax = axes[0]
    centres = np.linspace(0.05, 0.95, dcal.n_bins)
    ax.bar(
        centres,
        dcal.bin_counts,
        width=0.085,
        color=COLOR_LOW,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.axhline(
        dcal.expected,
        color=COLOR_HIGH,
        linestyle="--",
        linewidth=1.6,
        label="Expected under perfect calibration",
    )
    ax.set_xlabel(r"Predicted survival probability $\hat{S}(T_i \mid x_i)$")
    ax.set_ylabel("Patients (censoring-adjusted)")
    ax.set_title("(a) D-calibration")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(
        0.03,
        0.94,
        f"$\\chi^2_{{{dcal.n_bins - 1}}}$ = {dcal.chi2:.1f}, {_format_p(dcal.p_value)}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d0d0d0"),
    )

    # --- (b) 5-year reliability ------------------------------------------
    ax = axes[1]
    predicted = oof["surv_at_5y"].to_numpy()
    time, event = oof["time"].to_numpy(), oof["event"].to_numpy()

    deciles = pd.qcut(predicted, 10, labels=False, duplicates="drop")
    xs, ys, los, his = [], [], [], []
    for d in np.unique(deciles):
        mask = deciles == d
        km = KaplanMeierFitter().fit(time[mask], event[mask])
        observed = float(km.predict(FIVE_YEARS_MONTHS))
        ci = km.confidence_interval_survival_function_
        row = ci.iloc[np.searchsorted(ci.index.values, FIVE_YEARS_MONTHS, side="right") - 1]
        xs.append(predicted[mask].mean())
        ys.append(observed)
        los.append(float(row.iloc[0]))
        his.append(float(row.iloc[1]))

    xs, ys = np.asarray(xs), np.asarray(ys)
    ax.plot([0, 1], [0, 1], color=COLOR_NEUTRAL, linestyle=":", linewidth=1.5, label="Ideal")
    ax.errorbar(
        xs,
        ys,
        yerr=[ys - np.asarray(los), np.asarray(his) - ys],
        fmt="o",
        color=COLOR_LOW,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.0,
        elinewidth=1.2,
        capsize=3,
        label="Risk deciles (Kaplan-Meier)",
    )
    lo_lim = min(xs.min(), ys.min()) - 0.08
    ax.set_xlim(lo_lim, 1.0)
    ax.set_ylim(lo_lim, 1.0)
    ax.set_xlabel("Predicted 5-year survival")
    ax.set_ylabel("Observed 5-year survival")
    ax.set_title("(b) 5-year reliability by risk decile")
    ax.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def plot_cindex_by_uncertainty(results: dict, out_path: Path) -> None:
    """Discrimination and Brier score within terciles of epistemic uncertainty.

    Two metrics because they answer the same question from opposite ends: the
    C-index is a ranking measure whose value depends on how much risk spread a
    subgroup happens to contain, while the Brier score is a per-patient proper
    scoring rule. Both agreeing is what makes the trend believable.
    """
    table = results["uncertainty_bins"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))

    x = np.arange(len(table))
    labels = [
        f"{row.uncertainty_bin.capitalize()}\n(n = {row.n})" for row in table.itertuples()
    ]

    # --- (a) discrimination ----------------------------------------------
    ax = axes[0]
    heights = table["cindex"].to_numpy()
    yerr = np.vstack(
        [heights - table["ci_low"].to_numpy(), table["ci_high"].to_numpy() - heights]
    )
    ax.bar(
        x,
        heights,
        width=0.62,
        color=SEQUENTIAL_BLUES[: len(table)],
        edgecolor="white",
        linewidth=1.5,
        yerr=yerr,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#5a5a5a"},
    )
    ax.axhline(0.5, color=COLOR_NEUTRAL, linestyle=":", linewidth=1.4)
    ax.text(-0.42, 0.506, "chance", fontsize=8, color="#5a5a5a", ha="left")
    ax.axhline(
        results["bayesian"]["cindex"],
        color=COLOR_HIGH,
        linestyle="--",
        linewidth=1.5,
        label=f"Whole cohort ({results['bayesian']['cindex']:.3f})",
    )
    for xi, h in zip(x, heights):
        ax.text(xi, h + 0.012, f"{h:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Harrell's C-index")
    ax.set_title("(a) Discrimination")
    ax.set_ylim(0.45, max(table["ci_high"].max() + 0.05, 0.75))
    ax.legend(loc="lower left", fontsize=9)

    # --- (b) Brier score --------------------------------------------------
    ax = axes[1]
    brier = table["brier_5y"].to_numpy()
    ax.bar(
        x,
        brier,
        width=0.62,
        color=SEQUENTIAL_REDS[: len(table)],
        edgecolor="white",
        linewidth=1.5,
    )
    for xi, b in zip(x, brier):
        ax.text(xi, b + 0.004, f"{b:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("IPCW Brier score at 5 years")
    ax.set_title("(b) Predictive error (lower is better)")
    ax.set_ylim(0, brier.max() * 1.25)

    for ax in axes:
        ax.set_xlabel("Epistemic uncertainty tercile (risk-adjusted $\\sigma$)")

    fig.suptitle(
        "Predictions degrade where the model reports more uncertainty",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def plot_referral_curve(results: dict, out_path: Path) -> None:
    """C-index against the fraction of most-uncertain patients deferred."""
    curve = results["referral_curve"]
    raw = results["referral_curve_raw"]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    x = curve["referral_fraction"].to_numpy() * 100
    unc = curve["cindex_uncertainty_referral"].to_numpy()
    rnd = curve["cindex_random_referral"].to_numpy()
    sd = curve["cindex_random_sd"].to_numpy()

    ax.fill_between(x, rnd - sd, rnd + sd, color=COLOR_NEUTRAL, alpha=0.18, linewidth=0)
    ax.plot(
        x,
        rnd,
        color=COLOR_NEUTRAL,
        linewidth=2.0,
        linestyle="--",
        label="Random referral (control)",
    )
    ax.plot(
        raw["referral_fraction"].to_numpy() * 100,
        raw["cindex_uncertainty_referral"].to_numpy(),
        color=COLOR_RAW,
        linewidth=2.0,
        linestyle=":",
        marker="s",
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=r"Refer on raw $\sigma$",
    )
    ax.plot(
        x,
        unc,
        color=COLOR_LOW,
        linewidth=2.2,
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=r"Refer on risk-adjusted $\sigma$",
    )

    # Open a clear band above every series so the callout cannot land on top of
    # a curve or the control band.
    y_max = max(unc.max(), rnd.max() + sd.max(), raw["cindex_uncertainty_referral"].max())
    y_min = min(unc.min(), rnd.min() - sd.min(), raw["cindex_uncertainty_referral"].min())
    span = y_max - y_min
    ax.set_ylim(y_min - 0.05 * span, y_max + 0.26 * span)

    ref = results["referral"]
    pct = ref["fraction"] * 100
    ax.annotate(
        f"{pct:.0f}% deferred\nC = {ref['cindex']:.3f} ({ref['gain']:+.3f})",
        xy=(pct, ref["cindex"]),
        xytext=(pct, y_max + 0.16 * span),
        ha="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#5a5a5a", linewidth=1.1, shrinkB=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d0d0d0"),
    )

    ax.set_xlabel("Most uncertain patients deferred to a clinician (%)")
    ax.set_ylabel("C-index on retained patients")
    ax.set_title("Referral works only after uncertainty is decoupled from risk")
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def plot_shap_summary(
    model,
    X: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    table_path: Path,
    clinical_names: list[str] | None = None,
    full_table_path: Path | None = None,
    top_n: int = 20,
    n_background: int = 200,
    n_explain: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """SHAP beeswarm over the top ``top_n`` genes of the final model.

    Clinical covariates are ranked alongside the genes in the full table but
    kept out of the beeswarm: they dominate the top of the list, and letting
    them occupy the panel would leave room for only a handful of genes.
    """
    import shap
    import torch

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(X.shape[0], size=min(n_background, X.shape[0]), replace=False)
    explain_idx = rng.choice(X.shape[0], size=min(n_explain, X.shape[0]), replace=False)

    model.eval()  # attribution is for the posterior-mean model, so dropout is off
    background = torch.as_tensor(X[background_idx], dtype=torch.float32)
    to_explain = torch.as_tensor(X[explain_idx], dtype=torch.float32)

    # The network returns a flat risk vector; SHAP indexes outputs as (n, n_out).
    class _RiskOutput(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x).unsqueeze(-1)

    explainer = shap.GradientExplainer(_RiskOutput(model), background)
    shap_values = explainer.shap_values(to_explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values).reshape(len(explain_idx), -1)

    clinical = set(clinical_names or [])
    ranking = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_type": ["clinical" if f in clinical else "gene" for f in feature_names],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    if full_table_path is not None:
        ranking.to_csv(full_table_path, index=False)
        logger.info(f"wrote {full_table_path}")

    genes = ranking[ranking["feature_type"] == "gene"].copy()
    genes.insert(1, "gene_rank", np.arange(1, len(genes) + 1))
    genes.head(200).to_csv(table_path, index=False)
    logger.info(f"wrote {table_path}")

    top_idx = genes.index[:top_n].to_numpy()
    set_style()
    fig = plt.figure(figsize=(7.6, 7.4))
    shap.summary_plot(
        shap_values[:, top_idx],
        features=X[explain_idx][:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        max_display=top_n,
        show=False,
        plot_size=None,
        color_bar_label="Standardised expression",
    )
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on predicted log hazard)")

    # The caveat sits between the title and the panel. Below the axes it would
    # land on top of the x-label, which shap positions itself.
    if clinical:
        best_gene_rank = int(genes.iloc[0]["rank"])
        ax.set_title(f"Top {top_n} genes driving predicted risk", fontsize=12, pad=30)
        ax.text(
            0.5,
            1.012,
            f"{best_gene_rank - 1} clinical covariates outrank the leading gene — "
            "full ranking in results/tables/feature_importance.csv",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#5a5a5a",
        )
    else:
        ax.set_title(f"Top {top_n} genes driving predicted risk", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return ranking


def write_tables(results: dict, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    performance_table(results).to_csv(tables_dir / "performance_metrics.csv", index=False)
    results["uncertainty_bins"].to_csv(tables_dir / "cindex_by_uncertainty.csv", index=False)
    results["uncertainty_bins_raw"].to_csv(
        tables_dir / "cindex_by_uncertainty_raw_sigma.csv", index=False
    )
    results["referral_curve"].to_csv(tables_dir / "referral_curve.csv", index=False)
    results["referral_curve_raw"].to_csv(
        tables_dir / "referral_curve_raw_sigma.csv", index=False
    )

    strat = results["stratification"]
    pd.DataFrame(
        [
            {
                "logrank_p": strat.logrank_p,
                "hazard_ratio": strat.hazard_ratio,
                "hr_ci_low": strat.hr_ci[0],
                "hr_ci_high": strat.hr_ci[1],
                "median_os_low_risk": strat.median_survival["low_risk"],
                "median_os_high_risk": strat.median_survival["high_risk"],
            }
        ]
    ).to_csv(tables_dir / "risk_stratification.csv", index=False)
    logger.info(f"wrote result tables to {tables_dir}")


def summarise(results: dict) -> None:
    logger.info("=" * 72)
    logger.info("RESULTS")
    logger.info("=" * 72)
    for key, label in (
        ("baseline", "Clinical Cox baseline"),
        ("bayesian", "Bayesian MLP (MC Dropout)"),
        ("referral", "+ uncertainty referral"),
    ):
        r = results[key]
        dcal = r.get("dcal")
        line = (
            f"{label:<28} C = {r['cindex']:.4f} "
            f"[{r['cindex_ci'][0]:.4f}, {r['cindex_ci'][1]:.4f}]  "
            f"5-yr AUC = {r['auc_5y']:.4f}"
        )
        if dcal is not None:
            line += f"  D-cal p = {dcal.p_value:.3f}"
        if "gain" in r:
            line += f"  gain = {r['gain']:+.4f}"
        logger.info(line)
    strat = results["stratification"]
    logger.info(
        f"{'Risk stratification':<28} HR = {strat.hazard_ratio:.2f} "
        f"({strat.hr_ci[0]:.2f}-{strat.hr_ci[1]:.2f}), log-rank p = {strat.logrank_p:.3g}"
    )
    logger.info("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all figures and result tables.")
    parser.add_argument("--data-dir", default="data/brca_metabric")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-file", default="logs/figures.log")
    parser.add_argument("--skip-shap", action="store_true", help="skip the final-model refit")
    args = parser.parse_args()

    configure_logging(Path(args.log_file))
    set_style()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(results_dir / "oof_predictions.csv", index_col="sample_id")
    logger.info(f"loaded {len(oof)} out-of-fold predictions")

    results = evaluate(oof)
    summarise(results)
    write_tables(results, tables_dir)

    plot_km_curves(oof, results, figures_dir / "km_curve.png")
    plot_calibration(oof, results, figures_dir / "calibration_plot.png")
    plot_cindex_by_uncertainty(results, figures_dir / "cindex_by_uncertainty.png")
    plot_referral_curve(results, figures_dir / "referral_curve.png")

    if not args.skip_shap:
        config_path = results_dir / "run_config.json"
        cfg = TrainConfig(**json.loads(config_path.read_text())) if config_path.exists() else TrainConfig()
        cfg.hidden_dims = tuple(cfg.hidden_dims)
        data = load_metabric(args.data_dir)
        model, pre, X_all = fit_final_model(data, cfg)
        plot_shap_summary(
            model,
            X_all,
            pre.feature_names_,
            figures_dir / "shap_summary.png",
            tables_dir / "top_genes.csv",
            clinical_names=pre.clinical_names_,
            full_table_path=tables_dir / "feature_importance.csv",
        )

    logger.info("all figures and tables written")


if __name__ == "__main__":
    main()
