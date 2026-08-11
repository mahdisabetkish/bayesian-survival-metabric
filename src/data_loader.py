"""Loading and preprocessing of the METABRIC cohort.

The cBioPortal distribution ships tab-delimited files with four leading
comment lines before the real header, and the expression matrix is
genes-by-samples, so a fair amount of reshaping happens here.

Feature selection and standardisation are deliberately *not* done at load
time: both are fitted inside each cross-validation fold (see
:class:`FoldPreprocessor`) so that no information from held-out patients
leaks into the training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Clinical covariates used for the Cox baseline and appended to the
# expression block for the neural model.
NUMERIC_CLINICAL = [
    "AGE_AT_DIAGNOSIS",
    "TUMOR_SIZE",
    "LYMPH_NODES_EXAMINED_POSITIVE",
    "NPI",
    "GRADE",
]

# Categorical covariates encoded as a single indicator for the listed
# positive level; anything else (including missing) becomes 0.
BINARY_CLINICAL = {
    "ER_IHC": ("Positve", "ER_POSITIVE"),  # the source file carries this typo
    "HER2_SNP6": ("GAIN", "HER2_AMPLIFIED"),
    "PR_STATUS": ("Positive", "PR_POSITIVE"),
    "CHEMOTHERAPY": ("YES", "CHEMOTHERAPY"),
    "HORMONE_THERAPY": ("YES", "HORMONE_THERAPY"),
    "RADIO_THERAPY": ("YES", "RADIOTHERAPY"),
    "INFERRED_MENOPAUSAL_STATE": ("Post", "POST_MENOPAUSAL"),
}


@dataclass
class METABRICData:
    """Aligned expression, clinical and survival tables for one cohort."""

    expression: pd.DataFrame  # samples x genes, log2 microarray intensities
    clinical: pd.DataFrame  # samples x clinical covariates (raw, unscaled)
    time: np.ndarray  # overall survival, months
    event: np.ndarray  # 1 = death, 0 = censored
    sample_ids: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def gene_names(self) -> list[str]:
        return list(self.expression.columns)

    @property
    def clinical_names(self) -> list[str]:
        return list(self.clinical.columns)

    def structured_survival(self) -> np.ndarray:
        """Survival as a structured array, the format scikit-survival expects."""
        return np.array(
            list(zip(self.event.astype(bool), self.time)),
            dtype=[("event", "?"), ("time", "<f8")],
        )


def _read_cbioportal_table(path: Path) -> pd.DataFrame:
    """Read a cBioPortal clinical file, skipping its commented preamble."""
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


def _load_clinical(data_dir: Path) -> pd.DataFrame:
    patient = _read_cbioportal_table(data_dir / "data_clinical_patient.txt")
    sample = _read_cbioportal_table(data_dir / "data_clinical_sample.txt")
    merged = sample.merge(patient, on="PATIENT_ID", how="inner", suffixes=("", "_pat"))
    return merged.set_index("SAMPLE_ID")


def _load_expression(data_dir: Path) -> pd.DataFrame:
    """Return a samples-by-genes matrix of log2 intensities."""
    expr = pd.read_csv(
        data_dir / "data_mrna_illumina_microarray.txt",
        sep="\t",
        low_memory=False,
    )
    expr = expr.drop(columns=["Entrez_Gene_Id"], errors="ignore")
    expr = expr.dropna(subset=["Hugo_Symbol"])

    # A handful of Hugo symbols map to several probes. Keep the probe with the
    # largest variance, which is the one carrying signal.
    values = expr.drop(columns=["Hugo_Symbol"])
    variance = values.var(axis=1, skipna=True)
    order = variance.sort_values(ascending=False, kind="mergesort").index
    expr = expr.loc[order]
    expr = expr[~expr["Hugo_Symbol"].duplicated(keep="first")]

    expr = expr.set_index("Hugo_Symbol")
    expr = expr.apply(pd.to_numeric, errors="coerce")
    return expr.T.astype(np.float32)


def _encode_clinical(clinical: pd.DataFrame) -> pd.DataFrame:
    """Build the numeric clinical design matrix (missing values left as NaN)."""
    out = pd.DataFrame(index=clinical.index)

    for col in NUMERIC_CLINICAL:
        out[col] = pd.to_numeric(clinical.get(col), errors="coerce")

    for col, (positive_level, name) in BINARY_CLINICAL.items():
        if col not in clinical.columns:
            out[name] = np.nan
            continue
        series = clinical[col].astype("string").str.strip()
        indicator = (series.str.casefold() == positive_level.casefold()).astype(float)
        out[name] = indicator.where(series.notna(), np.nan)

    return out


def _cache_load(cache_file: Path) -> METABRICData | None:
    if not cache_file.exists():
        return None
    try:
        with np.load(cache_file, allow_pickle=True) as npz:
            return METABRICData(
                expression=pd.DataFrame(
                    npz["expression"],
                    index=npz["sample_ids"],
                    columns=npz["gene_names"],
                ),
                clinical=pd.DataFrame(
                    npz["clinical"],
                    index=npz["sample_ids"],
                    columns=npz["clinical_names"],
                ),
                time=npz["time"],
                event=npz["event"],
                sample_ids=npz["sample_ids"],
            )
    except (OSError, KeyError, ValueError):
        return None  # stale or truncated cache: fall through to a full parse


def _cache_save(cache_file: Path, data: METABRICData) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        expression=data.expression.to_numpy(dtype=np.float32),
        clinical=data.clinical.to_numpy(dtype=np.float64),
        gene_names=np.asarray(data.gene_names, dtype=object),
        clinical_names=np.asarray(data.clinical_names, dtype=object),
        sample_ids=data.sample_ids,
        time=data.time,
        event=data.event,
    )
    tmp.replace(cache_file)


def load_metabric(
    data_dir: str | Path,
    min_followup_months: float = 0.1,
    use_cache: bool = True,
) -> METABRICData:
    """Load METABRIC and align expression, clinical and overall-survival tables.

    Patients without expression data, without a follow-up time, or with a
    non-positive follow-up time are dropped — the last group cannot contribute
    to a partial likelihood.

    Parsing the 20k x 2k microarray text file takes about a minute, so the
    aligned tables are cached alongside the raw study.
    """
    data_dir = Path(data_dir)
    cache_file = data_dir.parent / "cache" / "metabric_aligned.npz"
    if use_cache:
        cached = _cache_load(cache_file)
        if cached is not None:
            return cached

    if not (data_dir / "data_mrna_illumina_microarray.txt").exists():
        raise FileNotFoundError(
            f"METABRIC files not found in {data_dir}. Run `bash data/download.sh` first."
        )

    clinical = _load_clinical(data_dir)
    expression = _load_expression(data_dir)

    shared = clinical.index.intersection(expression.index)
    clinical = clinical.loc[shared]
    expression = expression.loc[shared]

    time = pd.to_numeric(clinical["OS_MONTHS"], errors="coerce")
    status = clinical["OS_STATUS"].astype("string")
    event = status.str.startswith("1").astype("float")
    event = event.where(status.notna(), np.nan)

    keep = time.notna() & event.notna() & (time >= min_followup_months)
    clinical, expression = clinical.loc[keep], expression.loc[keep]
    time, event = time.loc[keep], event.loc[keep]

    # Drop genes that are entirely missing after alignment; impute the rest
    # with the gene mean, which is harmless for a microarray matrix that is
    # >99.9% complete.
    expression = expression.loc[:, expression.notna().any(axis=0)]
    expression = expression.fillna(expression.mean(axis=0))

    data = METABRICData(
        expression=expression,
        clinical=_encode_clinical(clinical),
        time=time.to_numpy(dtype=np.float64),
        event=event.to_numpy(dtype=np.float64),
        sample_ids=np.asarray(expression.index),
    )
    if use_cache:
        _cache_save(cache_file, data)
    return data


class FoldPreprocessor:
    """Fold-local gene selection, imputation and standardisation.

    Fitted on training patients only. Genes are ranked by variance, which is an
    unsupervised criterion, but it is still fitted inside the fold so the whole
    pipeline is honest about what the model was allowed to see.
    """

    def __init__(self, n_genes: int = 2000, use_clinical: bool = True):
        self.n_genes = n_genes
        self.use_clinical = use_clinical
        self.selected_genes_: list[str] = []
        self.clinical_names_: list[str] = []

    @property
    def feature_names_(self) -> list[str]:
        return self.selected_genes_ + self.clinical_names_

    @property
    def n_clinical_(self) -> int:
        return len(self.clinical_names_)

    def fit(self, data: METABRICData, idx: np.ndarray) -> "FoldPreprocessor":
        expr_train = data.expression.iloc[idx]
        variance = expr_train.var(axis=0)
        n_genes = min(self.n_genes, variance.size)
        self.selected_genes_ = list(
            variance.sort_values(ascending=False, kind="mergesort").index[:n_genes]
        )

        block = expr_train[self.selected_genes_].to_numpy(dtype=np.float64)
        self.gene_mean_ = block.mean(axis=0)
        self.gene_std_ = block.std(axis=0)
        self.gene_std_[self.gene_std_ < 1e-8] = 1.0

        if self.use_clinical:
            clin_train = data.clinical.iloc[idx]
            self.clinical_names_ = list(clin_train.columns)
            self.clinical_median_ = clin_train.median(axis=0, skipna=True).to_numpy()
            self.clinical_median_ = np.nan_to_num(self.clinical_median_, nan=0.0)
            filled = clin_train.to_numpy(dtype=np.float64)
            filled = np.where(np.isnan(filled), self.clinical_median_, filled)
            self.clinical_mean_ = filled.mean(axis=0)
            self.clinical_std_ = filled.std(axis=0)
            self.clinical_std_[self.clinical_std_ < 1e-8] = 1.0
        return self

    def transform(self, data: METABRICData, idx: np.ndarray) -> np.ndarray:
        block = data.expression.iloc[idx][self.selected_genes_].to_numpy(dtype=np.float64)
        block = (block - self.gene_mean_) / self.gene_std_
        if not self.use_clinical:
            return block.astype(np.float32)

        clin = data.clinical.iloc[idx].to_numpy(dtype=np.float64)
        clin = np.where(np.isnan(clin), self.clinical_median_, clin)
        clin = (clin - self.clinical_mean_) / self.clinical_std_
        return np.hstack([block, clin]).astype(np.float32)

    def fit_transform(self, data: METABRICData, idx: np.ndarray) -> np.ndarray:
        return self.fit(data, idx).transform(data, idx)

    def transform_clinical_only(self, data: METABRICData, idx: np.ndarray) -> pd.DataFrame:
        """Imputed, unscaled clinical block — the design matrix for the Cox baseline."""
        clin = data.clinical.iloc[idx].to_numpy(dtype=np.float64)
        clin = np.where(np.isnan(clin), self.clinical_median_, clin)
        return pd.DataFrame(clin, columns=self.clinical_names_, index=data.sample_ids[idx])
