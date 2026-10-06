"""
SVI and Laplace approximation inference backends for HDX rate fitting.
Drop-in research alternative to fit_rates() in inference.py.
"""
from __future__ import annotations
import time
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import (
    AutoNormal, AutoMultivariateNormal, AutoLaplaceApproximation,
)
import numpyro.optim as numpyro_optim

from .types import SVIConfig, PosteriorRates, PosteriorMergeFac, ProteinGroup, BackExchangeResult
from .spectra import theoretical_isotope_dist
from .isotopes import theoretical_dist_all_timepoints
from .inference import _build_model_inputs
from .model import recalc_timepoints_with_merge


_GUIDES = {
    'normal': AutoNormal,
    'mvn': AutoMultivariateNormal,
    'laplace': AutoLaplaceApproximation,
}


def fit_rates_svi(
    protein_group: ProteinGroup,
    backexchange_by_pH: dict[float, BackExchangeResult],
    config: SVIConfig,
    undeuterated_dist: np.ndarray,
    num_bins: int,
    reference_pH: float | None = None,
) -> tuple[PosteriorRates, PosteriorMergeFac | None, dict]:
    """
    SVI / Laplace approximation alternative to fit_rates().

    Returns:
        PosteriorRates — same structure as MCMC output.
            n_eff is set to num_posterior_samples (SVI samples are i.i.d.).
            r_hat is set to 1.0 (no chains concept in SVI).
        PosteriorMergeFac | None — merge factor posterior if multi-pH.
        dict — {'wall_time': float, 'final_elbo': float,
                'guide_type': str, 'rmse_total': float}
    """
    if reference_pH is None:
        reference_pH = min(protein_group.pHs, key=lambda pH: abs(pH - 6.0))

    guide_cls = _GUIDES.get(config.guide_type)
    if guide_cls is None:
        raise ValueError(
            f"Unknown guide_type '{config.guide_type}'. Choose from: {list(_GUIDES)}"
        )

    if config.num_steps < 1:
        raise ValueError("SVIConfig.num_steps must be >= 1")

    model_fn, model_kwargs, diag_tps, diag_bx, exp_dists = _build_model_inputs(
        protein_group, backexchange_by_pH, undeuterated_dist, num_bins, reference_pH
    )
    und = jnp.asarray(undeuterated_dist)

    guide = guide_cls(model_fn)
    optimizer = numpyro_optim.Adam(config.learning_rate)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    rng_key = random.PRNGKey(config.rng_seed)
    rng_key, sample_key = random.split(rng_key)

    t0 = time.perf_counter()
    svi_result = svi.run(rng_key, config.num_steps, progress_bar=False, **model_kwargs)
    wall_time = time.perf_counter() - t0

    params = svi_result.params
    final_elbo = float(-svi_result.losses[-1])

    predictive = Predictive(
        model_fn,
        guide=guide,
        num_samples=config.num_posterior_samples,
        params=params,
        return_sites=['rate', 'merge_fac'],
    )
    kwargs_no_obs = {**model_kwargs, 'obs_nonzero_flat': None}
    samples = predictive(sample_key, **kwargs_no_obs)

    # Sort rates slowest→fastest per sample (matches MCMC sort_posterior_rates convention)
    rate_samples = np.sort(np.array(samples['rate']), axis=1)  # (S, N)
    S = rate_samples.shape[0]

    posterior_rates = PosteriorRates(
        samples=rate_samples,
        mean=rate_samples.mean(axis=0),
        median=np.median(rate_samples, axis=0),
        std=rate_samples.std(axis=0),
        ci_5=np.percentile(rate_samples, 5, axis=0),
        ci_95=np.percentile(rate_samples, 95, axis=0),
        n_eff=np.full(rate_samples.shape[1], float(S)),
        r_hat=np.ones(rate_samples.shape[1]),
    )

    merge_factor = None
    if 'merge_fac' in samples:
        mf_samples = np.array(samples['merge_fac'])
        if mf_samples.ndim == 1:
            mf_samples = mf_samples[:, None]
        merge_factor = PosteriorMergeFac(
            samples=mf_samples,
            mean=mf_samples.mean(axis=0),
            std=mf_samples.std(axis=0),
            ci_5=np.percentile(mf_samples, 5, axis=0),
            ci_95=np.percentile(mf_samples, 95, axis=0),
        )
        eff_tps = np.array(recalc_timepoints_with_merge(
            model_kwargs['timepoints_list'],
            jnp.asarray(merge_factor.mean),
            model_kwargs['merge_fac_signs'],
        ))
    else:
        eff_tps = diag_tps

    pred_dists = np.array(theoretical_dist_all_timepoints(
        undeuterated_dist=und,
        timepoints=jnp.asarray(eff_tps),
        rates=jnp.exp(jnp.asarray(posterior_rates.mean)),
        inv_backexchange_array=jnp.subtract(1.0, jnp.asarray(diag_bx)),
        d2o_fraction=protein_group.d2o_fraction,
        d2o_purity=protein_group.d2o_purity,
        num_bins=num_bins,
    ))
    rmse_total = float(np.sqrt(np.mean((exp_dists.ravel() - pred_dists.ravel()) ** 2)))

    return posterior_rates, merge_factor, {
        'wall_time': wall_time,
        'final_elbo': final_elbo,
        'guide_type': config.guide_type,
        'rmse_total': rmse_total,
    }
