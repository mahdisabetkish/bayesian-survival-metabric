import numpy as np

from src.data_loader import FoldPreprocessor
from src.model import BreslowBaseline
from src.uncertainty import cumulative_dynamic_auc, d_calibration, harrell_cindex


def test_cindex_perfect_and_reversed():
    time = np.arange(1.0, 51.0)
    event = np.ones_like(time)
    assert harrell_cindex(-time, time, event) == 1.0  # shorter survival, higher risk
    assert harrell_cindex(time, time, event) == 0.0


def test_time_dependent_auc_perfect_separation():
    rng = np.random.default_rng(1)
    time = rng.uniform(1.0, 120.0, size=300)
    event = np.ones_like(time)
    risk = (time <= 60.0).astype(float) + rng.uniform(0, 0.1, size=300)
    assert cumulative_dynamic_auc(risk, time, event, t_eval=60.0) == 1.0


def test_dcalibration_accepts_calibrated_and_rejects_miscalibrated():
    rng = np.random.default_rng(2)
    time = rng.exponential(size=2000)
    event = np.ones_like(time)
    # Under the true model S(T) = exp(-T) is uniform on [0, 1].
    assert d_calibration(np.exp(-time), event).p_value > 0.05
    # A model that thinks everyone lives five times longer piles S(T) near 1.
    assert d_calibration(np.exp(-time / 5.0), event).p_value < 1e-6


def test_breslow_survival_is_monotone():
    rng = np.random.default_rng(3)
    risk = rng.normal(size=200)
    time = rng.exponential(scale=np.exp(-risk) * 50.0)
    event = (rng.uniform(size=200) < 0.7).astype(float)
    breslow = BreslowBaseline().fit(risk, time, event)

    grid = np.linspace(0.0, time.max(), 25)
    surv = breslow.survival_function(np.array([-1.0, 0.0, 1.0]), grid)
    assert np.all(np.diff(surv, axis=1) <= 1e-12)  # non-increasing in time
    assert np.all(surv[0] >= surv[1]) and np.all(surv[1] >= surv[2])  # lower risk, higher S


def test_fold_preprocessor_ignores_held_out_patients(cohort):
    train_idx = np.arange(0, 300)
    reference = FoldPreprocessor(n_genes=10).fit(cohort, train_idx)

    altered = cohort.expression.copy()
    altered.iloc[300:] = altered.iloc[300:] * 50.0 + 7.0  # only held-out rows change
    shifted = type(cohort)(altered, cohort.clinical, cohort.time, cohort.event, cohort.sample_ids)
    refit = FoldPreprocessor(n_genes=10).fit(shifted, train_idx)

    assert refit.selected_genes_ == reference.selected_genes_
    np.testing.assert_allclose(refit.gene_mean_, reference.gene_mean_)
    np.testing.assert_allclose(refit.clinical_mean_, reference.clinical_mean_)
