import numpy as np
import pandas as pd
import pytest

from src.data_loader import METABRICData


def make_cohort(n: int = 500, n_genes: int = 40, n_clinical: int = 12, seed: int = 0) -> METABRICData:
    """Synthetic cohort where genes 0-1 and clinical column 0 drive the hazard."""
    rng = np.random.default_rng(seed)
    expression = rng.normal(size=(n, n_genes))
    clinical = rng.normal(size=(n, n_clinical))
    log_hazard = 0.9 * expression[:, 0] - 0.9 * expression[:, 1] + 0.6 * clinical[:, 0]
    event_time = rng.exponential(scale=60.0 * np.exp(-log_hazard))
    censor_time = rng.uniform(10.0, 200.0, size=n)

    ids = np.array([f"S{i:04d}" for i in range(n)], dtype=object)
    return METABRICData(
        expression=pd.DataFrame(expression, index=ids, columns=[f"G{j}" for j in range(n_genes)]),
        clinical=pd.DataFrame(clinical, index=ids, columns=[f"C{j}" for j in range(n_clinical)]),
        time=np.minimum(event_time, censor_time),
        event=(event_time <= censor_time).astype(float),
        sample_ids=ids,
    )


@pytest.fixture(scope="session")
def cohort() -> METABRICData:
    return make_cohort()
