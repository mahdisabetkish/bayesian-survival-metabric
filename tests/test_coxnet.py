import numpy as np
import pytest

from src.coxnet_baseline import comparison_table, run_coxnet_cv
from src.train import TrainConfig, run_cross_validation
from src.uncertainty import harrell_cindex


@pytest.fixture(scope="module")
def runs(cohort):
    cfg = TrainConfig(n_genes=40, max_epochs=3, patience=3, n_mc_samples=3)
    coxnet_oof, folds = run_coxnet_cv(cohort, cfg, l1_ratios=(0.5,))
    main_oof, _ = run_cross_validation(cohort, cfg)
    return coxnet_oof, folds, main_oof


def test_coxnet_uses_the_same_folds_as_the_network(runs):
    coxnet_oof, _, main_oof = runs
    np.testing.assert_array_equal(coxnet_oof["fold"].to_numpy(), main_oof["fold"].to_numpy())


def test_coxnet_recovers_the_planted_signal(runs):
    coxnet_oof, folds, _ = runs
    assert not coxnet_oof[["risk_z", "surv_at_obs_time", "surv_at_5y"]].isna().any().any()
    pooled = harrell_cindex(
        coxnet_oof["risk_z"].to_numpy(), coxnet_oof["time"].to_numpy(), coxnet_oof["event"].to_numpy()
    )
    assert pooled > 0.7
    assert all(f.n_genes_selected >= 2 for f in folds)


def test_comparison_table_refuses_unpaired_folds(runs):
    coxnet_oof, _, main_oof = runs
    table = comparison_table(coxnet_oof, main_oof)
    assert list(table["model"]) == [
        "Clinical Cox baseline (12 covariates)",
        "CoxNet, 1,000 genes + clinical",
        "Bayesian MLP, 1,000 genes + clinical",
    ]

    shuffled = main_oof.copy()
    shuffled["fold"] = np.roll(shuffled["fold"].to_numpy(), 1)
    with pytest.raises(ValueError, match="not paired"):
        comparison_table(coxnet_oof, shuffled)
