"""Bayesian MLP survival model.

The network maps a patient's expression + clinical profile to a scalar log
hazard ratio and is trained with the Cox partial likelihood. Dropout is left
active at prediction time, which turns the network into an approximate
Bayesian posterior over risk scores (Gal & Ghahramani, 2016): the spread of
the Monte-Carlo samples is the model's epistemic uncertainty for that patient.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BayesianMLP(nn.Module):
    """Feed-forward risk model with dropout on every hidden layer.

    If ``n_clinical`` is given, the final ``n_clinical`` input columns are also
    fed through an undropped linear term that is added to the network output.
    A few hundred noisy expression features otherwise drown out a handful of
    strong clinical covariates; this shortcut lets the model fall back on a
    plain clinical Cox fit and use the deep branch for whatever the
    transcriptome adds on top of it.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 64),
        dropout: float = 0.3,
        batchnorm: bool = True,
        n_clinical: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.n_clinical = n_clinical

        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = width
        self.body = nn.Sequential(*layers)

        # No bias on the output: the Cox partial likelihood is invariant to a
        # constant shift in the risk score, so a bias term is unidentifiable.
        self.head = nn.Linear(prev, 1, bias=False)
        self.clinical_head = (
            nn.Linear(n_clinical, 1, bias=False) if n_clinical > 0 else None
        )
        self.apply(self._init_weights)
        if self.clinical_head is not None:
            # Start the shortcut at zero so training begins from the deep model
            # and grows the linear term only if it helps.
            nn.init.zeros_(self.clinical_head.weight)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.body(x))
        if self.clinical_head is not None:
            out = out + self.clinical_head(x[:, -self.n_clinical :])
        return out.squeeze(-1)

    def enable_mc_dropout(self) -> None:
        """Put the model in eval mode but keep the dropout layers stochastic.

        Batch-norm must stay in eval mode, otherwise the running statistics
        would be re-estimated from the test batch.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


def cox_partial_likelihood_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Negative log Cox partial likelihood with the Breslow tie correction.

    Averaged over the number of observed events so the value is comparable
    across batches with different numbers of deaths.
    """
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]

    # log sum_{j in risk set} exp(risk_j), where the risk set of patient i is
    # everyone still at risk at t_i — i.e. everyone earlier in this ordering.
    log_risk_set = torch.logcumsumexp(risk, dim=0)

    n_events = event.sum()
    if n_events < 1:
        return risk.sum() * 0.0
    return -torch.sum((risk - log_risk_set) * event) / n_events


@torch.no_grad()
def predict_mc(
    model: BayesianMLP,
    X: np.ndarray,
    n_samples: int = 100,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> np.ndarray:
    """Draw ``n_samples`` stochastic risk scores per patient.

    Returns an array of shape ``(n_samples, n_patients)``.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = model.to(device)
    model.enable_mc_dropout()
    tensor = torch.as_tensor(X, dtype=torch.float32, device=device)

    draws = np.empty((n_samples, tensor.shape[0]), dtype=np.float64)
    for s in range(n_samples):
        chunks = []
        for start in range(0, tensor.shape[0], batch_size):
            chunks.append(model(tensor[start : start + batch_size]))
        draws[s] = torch.cat(chunks).cpu().numpy()
    return draws


@torch.no_grad()
def predict_deterministic(
    model: BayesianMLP,
    X: np.ndarray,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Risk scores with dropout switched off — the posterior mean approximation."""
    model = model.to(device)
    model.eval()
    tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    chunks = [
        model(tensor[start : start + batch_size])
        for start in range(0, tensor.shape[0], batch_size)
    ]
    return torch.cat(chunks).cpu().numpy().astype(np.float64)


class BreslowBaseline:
    """Breslow estimator of the baseline cumulative hazard.

    Needed to turn a Cox risk score into an absolute survival probability,
    which the calibration analysis requires.
    """

    def __init__(self) -> None:
        self.times_: np.ndarray = np.array([])
        self.cum_hazard_: np.ndarray = np.array([])

    def fit(self, risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> "BreslowBaseline":
        # Centre the risk scores so the baseline corresponds to an average
        # patient rather than to a hypothetical patient with risk score 0.
        self.offset_ = float(np.mean(risk))
        hazard_ratio = np.exp(np.asarray(risk, dtype=np.float64) - self.offset_)
        return self._fit_hazard_ratio(hazard_ratio, time, event)

    def fit_posterior(
        self, draws: np.ndarray, time: np.ndarray, event: np.ndarray
    ) -> "BreslowBaseline":
        """Fit from MC-dropout draws, using the posterior mean *hazard ratio*.

        The quantity the Cox model multiplies the baseline hazard by is
        exp(f(x)), so the posterior estimate is the mean of exp over the draws,
        not exp of the mean. The two differ by a Jensen gap that grows with a
        patient's posterior spread, which is exactly the systematic error the
        calibration test is sensitive to.
        """
        draws = np.asarray(draws, dtype=np.float64)
        self.offset_ = float(draws.mean())
        return self._fit_hazard_ratio(np.exp(draws - self.offset_).mean(axis=0), time, event)

    def _fit_hazard_ratio(
        self, hazard_ratio: np.ndarray, time: np.ndarray, event: np.ndarray
    ) -> "BreslowBaseline":
        order = np.argsort(time, kind="mergesort")
        time, event, hazard_ratio = time[order], event[order], hazard_ratio[order]

        # Risk set at each time is everyone with follow-up >= that time.
        at_risk = np.cumsum(hazard_ratio[::-1])[::-1]

        uniq, first = np.unique(time, return_index=True)
        deaths = np.add.reduceat(event, first)
        denom = at_risk[first]

        increments = np.divide(deaths, denom, out=np.zeros_like(deaths), where=denom > 0)
        self.times_ = uniq
        self.cum_hazard_ = np.cumsum(increments)
        return self

    def survival_at(self, risk: np.ndarray, time: np.ndarray) -> np.ndarray:
        """S(t | x) evaluated per patient at that patient's own time point."""
        idx = np.searchsorted(self.times_, np.asarray(time, dtype=np.float64), side="right") - 1
        base = np.where(idx >= 0, self.cum_hazard_[np.clip(idx, 0, None)], 0.0)
        return np.exp(-base * np.exp(np.asarray(risk, dtype=np.float64) - self.offset_))

    def survival_function(self, risk: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Survival matrix of shape ``(n_patients, len(times))``."""
        idx = np.searchsorted(self.times_, np.asarray(times, dtype=np.float64), side="right") - 1
        base = np.where(idx >= 0, self.cum_hazard_[np.clip(idx, 0, None)], 0.0)
        hazard_ratio = np.exp(np.asarray(risk, dtype=np.float64) - self.offset_)
        return np.exp(-np.outer(hazard_ratio, base))

    def _baseline_at(self, time: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.times_, np.asarray(time, dtype=np.float64), side="right") - 1
        return np.where(idx >= 0, self.cum_hazard_[np.clip(idx, 0, None)], 0.0)

    def posterior_survival_at(self, draws: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Posterior predictive S(t | x): the survival curves are averaged, not the risks."""
        base = self._baseline_at(time)  # (n,)
        hazard_ratio = np.exp(np.asarray(draws, dtype=np.float64) - self.offset_)  # (S, n)
        return np.exp(-base[None, :] * hazard_ratio).mean(axis=0)

    def posterior_survival_function(
        self, draws: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Posterior predictive survival matrix of shape ``(n_patients, len(times))``."""
        base = self._baseline_at(times)  # (t,)
        hazard_ratio = np.exp(np.asarray(draws, dtype=np.float64) - self.offset_)  # (S, n)
        return np.exp(-hazard_ratio[:, :, None] * base[None, None, :]).mean(axis=0)
