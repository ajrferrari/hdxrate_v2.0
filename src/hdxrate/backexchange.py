from __future__ import annotations
import numpy as np
from scipy.optimize import fmin_powell, curve_fit
from scipy.special import expit
from scipy import odr

from .types import BackExchangeResult, TimepointCorrection, ProteinGroup
from .spectra import theoretical_isotope_dist


def _gen_temp_rates(sequence: str, rate_value: float = 1e4) -> np.ndarray:
    """High uniform rates for fully-deuterated theoretical distribution."""
    rates = np.full(len(sequence), rate_value, dtype=float)
    rates[:2] = 0.0
    for i, aa in enumerate(sequence):
        if aa == 'P':
            rates[i] = 0.0
    return rates


def _poisson_binomial_numpy(probs: np.ndarray) -> np.ndarray:
    """Numpy (non-JAX) PoiBin for use inside scipy optimizers."""
    n = len(probs)
    omega = 2 * np.pi / (n + 1)
    half_n = int(n / 2 + n % 2)
    chi = np.zeros(n + 1, dtype=complex)
    chi[0] = 1
    exp_value = np.exp(omega * np.arange(1, half_n + 1) * 1j)
    xy = 1 - probs + probs * exp_value[:, np.newaxis]
    argz_sum = np.arctan2(xy.imag, xy.real).sum(axis=1)
    d_value = np.exp(np.log(np.abs(xy)).sum(axis=1))
    chi[1:half_n + 1] = d_value * np.exp(argz_sum * 1j)
    chi[half_n + 1:n + 1] = np.conjugate(chi[1:n - half_n + 1][::-1])
    chi /= n + 1
    return np.fft.fft(chi).real


def _isotope_dist_numpy(
    undeuterated_dist: np.ndarray,
    timepoint: float,
    rates: np.ndarray,
    inv_backexchange: float,
    d2o_fraction: float,
    d2o_purity: float,
    num_bins: int,
) -> np.ndarray:
    """Numpy isotope distribution for use in the backexchange optimizer."""
    probs = (1.0 - np.exp(-rates * timepoint)) * d2o_fraction * d2o_purity * inv_backexchange
    pmf = _poisson_binomial_numpy(probs)
    conv = np.convolve(pmf, undeuterated_dist)[:num_bins]
    return conv / conv.max()


def fit_protein_backexchange(
    sequence: str,
    final_tp_dist: np.ndarray,   # normalized mass dist at the last (most saturated) timepoint
    timepoints: np.ndarray,
    d2o_fraction: float,
    d2o_purity: float,
    bkex_tp: float = 1e9,
) -> BackExchangeResult:
    """
    Optimize a scalar backexchange fraction by fitting a fully-deuterated
    theoretical distribution to the observed final-timepoint distribution.

    Uses Powell optimizer with logit parameterization (v1 expit trick):
    optimization variable x → inv_backexchange = expit(x) = 1 - backexchange
    """
    rates = _gen_temp_rates(sequence)
    num_bins = len(final_tp_dist)
    undeuterated = theoretical_isotope_dist(sequence, num_bins)

    def objective(x):
        inv_bx = float(expit(np.atleast_1d(x)[0]))
        thr = _isotope_dist_numpy(undeuterated, bkex_tp, rates, inv_bx,
                                   d2o_fraction, d2o_purity, num_bins)
        return float(np.sqrt(np.mean((final_tp_dist - thr) ** 2)))

    opt = fmin_powell(objective, x0=np.array([0.05]), disp=False)
    backexchange_value = float(1.0 - expit(np.atleast_1d(opt)[0]))

    if backexchange_value == 1.0:
        backexchange_value = 0.98

    inv_bx = 1.0 - backexchange_value
    thr_sat = _isotope_dist_numpy(undeuterated, bkex_tp, rates, inv_bx,
                                   d2o_fraction, d2o_purity, num_bins)
    fit_rmse = float(np.sqrt(np.mean((final_tp_dist - thr_sat) ** 2)))

    backexchange_array = np.full(len(timepoints), backexchange_value)

    return BackExchangeResult(
        backexchange_value=backexchange_value,
        backexchange_array=backexchange_array,
        fit_rmse=fit_rmse,
        theoretical_dist_at_saturation=thr_sat,
    )


def apply_timepoint_correction(
    backexchange_value: float,
    timepoints: np.ndarray,
    correction: TimepointCorrection | None,
) -> np.ndarray:
    """
    Build per-timepoint backexchange array.

    If no correction, returns a constant array of backexchange_value.
    Otherwise uses v1 gen_corr_backexchange formula:
        bkexch[t] = ((1 - bkexch_final) / (-1)) * mass_rate[t] + bkexch_final
    which linearly interpolates between bkexch_final (at mass_rate=0)
    and 1.0 (at mass_rate=-1).
    """
    if correction is None:
        return np.full(len(timepoints), backexchange_value)

    tp_to_correction = dict(zip(correction.timepoints, correction.correction_values))
    mass_rate_array = np.array([tp_to_correction.get(tp, 0.0) for tp in timepoints])

    bkexch_array = (((1.0 - backexchange_value) / (-1.0)) * mass_rate_array) + backexchange_value
    return bkexch_array


def _fit_gaussian_centroid(dist: np.ndarray) -> float:
    """Fit a Gaussian to a distribution and return its centroid."""
    xdata = np.arange(len(dist))
    ymax = dist.max()
    if ymax == 0:
        return float(np.sum(xdata * dist))

    peak_idx = int(np.argmax(dist))
    norm = dist / ymax
    width_est = max(len(norm[norm > 0.7]), 1)
    p0 = [0.0, ymax, float(peak_idx), float(width_est)]

    try:
        def gauss(x, y0, A, xc, w):
            return y0 + A * np.exp(-((x - xc) ** 2) / (2 * w ** 2))

        popt, _ = curve_fit(gauss, xdata, dist, p0=p0, maxfev=10000)
        if popt[2] > 0 and 0 < popt[3] < len(dist):
            return float(popt[2])
    except RuntimeError:
        pass

    return float(np.sum(xdata * dist) / (dist.sum() + 1e-12))


def compute_timepoint_correction(
    protein_groups: list[ProteinGroup],
    backexchanges: dict[str, BackExchangeResult],  # protein_id → BackExchangeResult
    rate_threshold: float = 0.08,
    min_num_points: int = 3,
) -> TimepointCorrection:
    """
    Cross-protein timepoint-specific backexchange correction.

    For each protein, computes mass_diff_rate[t] = (centroid[t] - centroid[-1]) / centroid[-1].
    Pools rates per timepoint, filters outliers, takes median.
    Proteins not in backexchanges are skipped.

    Preserves v1 gen_bkexch_correction_ logic.
    """
    from collections import OrderedDict
    tp_rates: dict[float, list[float]] = OrderedDict()

    for pg in protein_groups:
        # Use first available pH (typically pH6)
        pH = pg.pHs[0]
        tps = pg.timepoints_by_pH[pH]
        dists = pg.distributions_by_pH[pH]

        centroids = np.array([_fit_gaussian_centroid(d) for d in dists])
        ref = centroids[-1]
        if ref == 0:
            continue

        mass_diff_rates = (centroids - ref) / ref

        for tp, rate in zip(tps, mass_diff_rates):
            tp_rates.setdefault(float(tp), []).append(rate)

    # Filter and compute median per timepoint
    timepoint_list: list[float] = [0.0]
    correction_list: list[float] = [-0.9]  # undeuterated reference

    for tp, rates_list in tp_rates.items():
        if tp == 0:
            continue
        filtered = [r for r in rates_list if abs(r) <= rate_threshold]
        if len(filtered) < min_num_points:
            filtered = rates_list  # fall back to all if too few pass filter

        if filtered:
            timepoint_list.append(tp)
            correction_list.append(float(np.median(filtered)))

    tps_arr = np.array(timepoint_list)
    corr_arr = np.array(correction_list)

    # Replace NaN with mean of non-edge values
    if np.any(np.isnan(corr_arr)):
        interior_mean = float(np.nanmean(corr_arr[1:-1])) if len(corr_arr) > 2 else 0.0
        corr_arr = np.where(np.isnan(corr_arr), interior_mean, corr_arr)

    order = np.argsort(tps_arr)
    return TimepointCorrection(
        timepoints=tps_arr[order],
        correction_values=corr_arr[order],
    )


def check_saturation(
    distributions: np.ndarray,   # (T, B) — sorted by timepoint
    min_timepoints: int = 3,
    mass_rate_threshold: float = 1.0,
) -> bool:
    """
    Check if the protein reached full deuteration.
    Requires at least min_timepoints timepoints; compares the Gaussian centroid
    at the last timepoint vs 5th-from-last (or earliest available).
    Returns True if the centroid has plateaued within mass_rate_threshold Da.
    """
    n = distributions.shape[0]
    if n < min_timepoints:
        return False
    c_last = _fit_gaussian_centroid(distributions[-1])
    c_ref = _fit_gaussian_centroid(distributions[max(0, n - 5)])
    return abs(c_last - c_ref) < mass_rate_threshold


def estimate_backexchange_from_high_pH(
    low_backexchange: np.ndarray,   # shape (N,) — lower pH
    high_backexchange: np.ndarray,  # shape (N,) — higher pH
    low_saturated: np.ndarray,      # bool (N,)
    high_saturated: np.ndarray,     # bool (N,)
    backexchange_lb: float = 0.0,
    backexchange_ub: float = 1.0,
) -> np.ndarray:
    """
    Rescue low-pH backexchange for unsaturated proteins using ODR linear
    regression fitted on proteins that saturated at BOTH pH conditions.

    Returns corrected low_backexchange array; unsaturated entries are replaced
    with values inferred by inverting the regression.
    """
    both_saturated = low_saturated & high_saturated
    in_bounds = (
        (low_backexchange >= backexchange_lb) & (low_backexchange <= backexchange_ub) &
        (high_backexchange >= backexchange_lb) & (high_backexchange <= backexchange_ub)
    )
    mask = both_saturated & in_bounds

    corrected = low_backexchange.copy()

    if mask.sum() < 2:
        return corrected

    x = low_backexchange[mask]
    y = high_backexchange[mask]

    odr_data = odr.Data(x=x, y=y)
    odr_fit = odr.ODR(data=odr_data, model=odr.unilinear).run()
    slope, intercept = odr_fit.beta[0], odr_fit.beta[1]

    if abs(slope) < 1e-9:
        return corrected

    for i in range(len(corrected)):
        if not low_saturated[i] and high_saturated[i]:
            corrected[i] = (high_backexchange[i] - intercept) / slope

    return corrected


def estimate_ph6_backexchange_from_ph9(
    ph6_backexchange: np.ndarray,
    ph9_backexchange: np.ndarray,
    ph6_saturated: np.ndarray,
    ph9_saturated: np.ndarray,
    backexchange_lb: float = 0.0,
    backexchange_ub: float = 1.0,
) -> np.ndarray:
    """Backwards-compatible alias for estimate_backexchange_from_high_pH."""
    return estimate_backexchange_from_high_pH(
        ph6_backexchange, ph9_backexchange,
        ph6_saturated, ph9_saturated,
        backexchange_lb, backexchange_ub,
    )
