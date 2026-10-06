"""
NumPyro generative models for HDX rate fitting.
These are plain Python functions that numpyro traces — no class wrapping.
Only this module and isotopes.py use JAX.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .isotopes import theoretical_dist_all_timepoints


def hdx_rate_model(
    num_rates: int,
    undeuterated_dist: jnp.ndarray,
    timepoints: jnp.ndarray,
    backexchange_array: jnp.ndarray,
    d2o_fraction: float,
    d2o_purity: float,
    num_bins: int,
    obs_nonzero_flat: jnp.ndarray,
    nonzero_indices: jnp.ndarray,
) -> None:
    """
    Single-pH numpyro model. Preserves v1 rate_fit_model_norm_priors exactly.

    Priors:
      slow_rate ~ TruncatedNormal(loc=-7, scale=10, low=-15, high=10)
      rate_centers = linspace(slow_rate, 10, num_rates)
      rate[i] ~ Normal(rate_centers[i], scale=5)
      sigma ~ Exponential(1)

    Likelihood:
      theoretical dists (via PoiBin) flattened at nonzero_indices ~ Normal(obs, sigma)
    """
    slow_rate = numpyro.sample(
        'slow_rate',
        dist.TruncatedNormal(loc=-7.0, scale=10.0, low=-15.0, high=10.0),
    )
    rate_centers = jnp.linspace(slow_rate, 10.0, num_rates)

    with numpyro.plate('rates', num_rates):
        rates_ = numpyro.sample('rate', dist.Normal(loc=rate_centers, scale=5.0))

    thr_dists = theoretical_dist_all_timepoints(
        undeuterated_dist=undeuterated_dist,
        timepoints=timepoints,
        rates=jnp.exp(rates_),
        inv_backexchange_array=jnp.subtract(1.0, backexchange_array),
        d2o_fraction=d2o_fraction,
        d2o_purity=d2o_purity,
        num_bins=num_bins,
    )

    flat_thr = jnp.concatenate(thr_dists)
    flat_thr_nonzero = flat_thr[nonzero_indices]

    sigma = numpyro.sample('sigma', dist.Exponential(1.0))

    with numpyro.plate('bins', len(flat_thr_nonzero)):
        numpyro.sample(
            'bin_preds',
            dist.Normal(loc=flat_thr_nonzero, scale=sigma),
            obs=obs_nonzero_flat,
        )


@jax.jit
def recalc_timepoints_with_merge(
    timepoints_list: list[jnp.ndarray],
    merge_facs: jnp.ndarray,
    merge_fac_signs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Concatenate reference timepoints with scaled non-reference timepoints.

    Effective timepoint for pH_i = actual_tp_i * 10^(sign_i * merge_fac_i).
      sign = +1 for pHs above reference (faster exchange, scale up)
      sign = -1 for pHs below reference (slower exchange, scale down)
    """
    result = [timepoints_list[0]]
    for i, (mf, sign) in enumerate(zip(merge_facs, merge_fac_signs)):
        result.append(timepoints_list[i + 1] * (10.0 ** (sign * mf)))
    return jnp.concatenate(result)


def hdx_rate_model_with_merge(
    num_rates: int,
    undeuterated_dist: jnp.ndarray,
    timepoints_list: list[jnp.ndarray],  # [ref_tps, pH_1_tps, ...]
    num_merge_facs: int,
    backexchange_array: jnp.ndarray,     # concatenated across pHs in same order
    d2o_fraction: float,
    d2o_purity: float,
    num_bins: int,
    obs_nonzero_flat: jnp.ndarray,
    nonzero_indices: jnp.ndarray,
    merge_fac_locs: jnp.ndarray,         # prior mean for each merge factor
    merge_fac_signs: jnp.ndarray,        # +1 above reference pH, -1 below
) -> None:
    """
    Multi-pH numpyro model (2 or more pH conditions).

    Prior per merge factor j:
      merge_fac[j] ~ TruncatedNormal(loc=merge_fac_locs[j], scale=0.05, low=0)
    where merge_fac_locs[j] = |pH_j - pH_ref| (auto-derived, e.g. 3.0 for pH9 vs pH6).

    Effective timepoints: ref_tps concatenated with
      pH_j_tps * 10^(sign_j * merge_fac_j)
    """
    slow_rate = numpyro.sample(
        'slow_rate',
        dist.TruncatedNormal(loc=-7.0, scale=10.0, low=-15.0, high=10.0),
    )
    rate_centers = jnp.linspace(slow_rate, 10.0, num_rates)

    with numpyro.plate('rates', num_rates):
        rates_ = numpyro.sample('rate', dist.Normal(loc=rate_centers, scale=5.0))

    with numpyro.plate('merge_facs', num_merge_facs):
        merge_facs_ = numpyro.sample(
            'merge_fac',
            dist.TruncatedNormal(loc=merge_fac_locs, scale=0.05, low=0.0),
        )

    concat_tps = recalc_timepoints_with_merge(timepoints_list, merge_facs_, merge_fac_signs)

    thr_dists = theoretical_dist_all_timepoints(
        undeuterated_dist=undeuterated_dist,
        timepoints=concat_tps,
        rates=jnp.exp(rates_),
        inv_backexchange_array=jnp.subtract(1.0, backexchange_array),
        d2o_fraction=d2o_fraction,
        d2o_purity=d2o_purity,
        num_bins=num_bins,
    )

    flat_thr = jnp.concatenate(thr_dists)
    flat_thr_nonzero = flat_thr[nonzero_indices]

    sigma = numpyro.sample('sigma', dist.Exponential(1.0))

    with numpyro.plate('bins', len(flat_thr_nonzero)):
        numpyro.sample(
            'bin_preds',
            dist.Normal(loc=flat_thr_nonzero, scale=sigma),
            obs=obs_nonzero_flat,
        )
