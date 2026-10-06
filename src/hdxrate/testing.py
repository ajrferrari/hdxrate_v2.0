"""
Mock data generator for testing and validation of the HDX rate fitting pipeline.
Generates synthetic InputRows from known ground-truth rates using the same
forward model as the inference, enabling quantitative recovery assessment.
"""
from __future__ import annotations
import numpy as np
from .types import InputRow, HDXResult
from .spectra import theoretical_isotope_dist
from .backexchange import _gen_temp_rates, _isotope_dist_numpy

PROTON_MASS = 1.007276


def generate_mock_rows(
    sequence: str,
    true_log_rates: np.ndarray,                      # shape (N_exchangeable,) — ground truth log-rates
    timepoints: np.ndarray,                           # e.g. [10, 60, 300, 1800, 7200, 86400] seconds
    pHs: list[float] | None = None,                  # [6.0], [6.0, 9.0], or [5.0, 6.0, 9.0]; defaults to [6.0]
    charge: int = 3,
    d2o_fraction: float = 0.95,
    d2o_purity: float = 0.95,
    temperature: float = 295.0,
    true_backexchange: float = 0.20,
    true_merge_factors: list[float] | float = 3.0,   # one per non-reference pH (reference = closest to 6.0)
    noise_sigma: float = 0.01,
    num_bins: int = 30,
    rng_seed: int = 42,
) -> list[InputRow]:
    """
    Generate synthetic InputRows from known ground-truth exchange rates.

    The forward model is identical to the inference model:
    - Exchange probabilities via (1 - exp(-k*t)) * d2o_fraction * d2o_purity * (1-backexchange)
    - Isotope distribution via Poisson-binomial convolved with undeuterated spectrum
    - Optional Gaussian noise on intensities

    For multi-pH, the reference is the pH closest to 6.0. For each other pH:
      effective_tp = actual_tp * 10^(sign * merge_factor)
    where sign = +1 if pH > reference (faster exchange) and -1 if pH < reference (slower).
    true_merge_factors order matches sorted non-reference pHs (same as inference).

    Returns InputRows with synthetic m/z = mass_grid/charge + PROTON_MASS.
    """
    if pHs is None:
        pHs = [6.0]

    reference_pH = min(pHs, key=lambda pH: abs(pH - 6.0))
    other_pHs = sorted(pH for pH in pHs if pH != reference_pH)

    if other_pHs:
        if isinstance(true_merge_factors, (int, float)):
            true_merge_factors = [float(true_merge_factors)]
        if len(true_merge_factors) != len(other_pHs):
            raise ValueError(
                f"true_merge_factors has {len(true_merge_factors)} values but there are "
                f"{len(other_pHs)} non-reference pHs: {other_pHs}"
            )

    rng = np.random.default_rng(rng_seed)
    undeuterated = theoretical_isotope_dist(sequence, num_bins).astype(np.float64)

    # Expand true log-rates to full-sequence linear rates array
    rates_full = _gen_temp_rates(sequence)
    exchangeable_idx = np.where(rates_full > 0)[0]
    if len(true_log_rates) != len(exchangeable_idx):
        raise ValueError(
            f"true_log_rates has {len(true_log_rates)} values but sequence has "
            f"{len(exchangeable_idx)} exchangeable residues"
        )
    rates_full[exchangeable_idx] = np.exp(true_log_rates)

    rows: list[InputRow] = []

    for pH in pHs:
        if pH == reference_pH:
            eff_timepoints = timepoints
        else:
            idx = other_pHs.index(pH)
            mf = true_merge_factors[idx]
            sign = 1.0 if pH > reference_pH else -1.0
            scale = 10.0 ** (sign * mf)
            eff_timepoints = timepoints * scale

        for tp, eff_tp in zip(timepoints, eff_timepoints):
            inv_bx = 1.0 - true_backexchange
            dist = _isotope_dist_numpy(
                undeuterated, eff_tp, rates_full, inv_bx,
                d2o_fraction, d2o_purity, num_bins,
            )

            # Add Gaussian noise, clip to ≥0, re-normalize to max=1
            noisy = dist + rng.normal(0, noise_sigma, size=num_bins)
            noisy = np.clip(noisy, 0, None)
            if noisy.max() > 0:
                noisy /= noisy.max()

            # Fake m/z: synthetic base mass for this sequence (rough average)
            base_mass = 500.0 + len(sequence) * 110.0  # crude approximate monoisotopic mass
            mass_grid = np.arange(base_mass, base_mass + num_bins, dtype=float)
            mz = (mass_grid + charge * PROTON_MASS) / charge

            rows.append(InputRow(
                protein_id='mock_protein',
                sequence=sequence,
                timepoint=float(tp),
                mz=mz,
                intensities=noisy,
                charge=charge,
                pH=pH,
                d2o_fraction=d2o_fraction,
                d2o_purity=d2o_purity,
                temperature=temperature,
            ))

    return rows


def recovery_summary(
    result: HDXResult,
    true_log_rates: np.ndarray,
) -> dict:
    """
    Compare posterior mean rates to ground-truth log-rates.

    Returns dict with:
        mae: mean absolute error (log-rate units)
        rmse: root mean squared error (log-rate units)
        coverage_90pct: fraction of true rates inside the 90% credible interval
        n_rates: number of rates compared
    """
    posterior_mean = result.posterior_rates.mean
    ci_5 = result.posterior_rates.ci_5
    ci_95 = result.posterior_rates.ci_95

    n = min(len(true_log_rates), len(posterior_mean))
    true = true_log_rates[:n]
    pred = posterior_mean[:n]
    lo = ci_5[:n]
    hi = ci_95[:n]

    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    coverage = float(np.mean((true >= lo) & (true <= hi)))

    return {
        'mae': mae,
        'rmse': rmse,
        'coverage_90pct': coverage,
        'n_rates': n,
    }
