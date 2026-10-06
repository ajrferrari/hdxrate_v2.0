"""
MCMC runner, chain diagnostics, and posterior summary for HDX rate fitting.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from jax import random
from numpyro.infer import NUTS, MCMC

from .types import (
    MCMCConfig, PosteriorRates, PosteriorMergeFac, ChainDiagnostics,
    ProteinGroup, BackExchangeResult,
)
from .isotopes import theoretical_dist_all_timepoints
from .model import hdx_rate_model, hdx_rate_model_with_merge, recalc_timepoints_with_merge
from .spectra import theoretical_isotope_dist


def _num_exchange_rates(sequence: str) -> int:
    """Count exchangeable residues (excludes first two and prolines)."""
    count = 0
    for i, aa in enumerate(sequence):
        if i < 2:
            continue
        if aa != 'P':
            count += 1
    return count


def _flat_nonzero(distributions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a (T, B) array and return nonzero values + their indices."""
    flat = distributions.ravel()
    indices = np.nonzero(flat)[0]
    return flat[indices], indices


def sort_posterior_rates(posterior_by_chain: dict) -> dict:
    """
    Sort rate samples within each sample slowest→fastest for all chains.
    Preserves v1 sort_posterior_rates_in_samples exactly.
    """
    if 'rate' in posterior_by_chain:
        chains = posterior_by_chain['rate']
        sorted_chains = jnp.array([jnp.sort(c, axis=1) for c in chains])
        posterior_by_chain = dict(posterior_by_chain)
        posterior_by_chain['rate'] = sorted_chains
    return posterior_by_chain


def _compute_chain_rmse(
    chain_rate_samples: np.ndarray,   # (S, N) log-rates for one chain
    undeuterated_dist: jnp.ndarray,
    timepoints: np.ndarray,
    backexchange_array: np.ndarray,
    d2o_fraction: float,
    d2o_purity: float,
    num_bins: int,
    exp_distributions: np.ndarray,    # (T, B)
) -> float:
    mean_rates = np.mean(chain_rate_samples, axis=0)
    thr = np.array(theoretical_dist_all_timepoints(
        undeuterated_dist=undeuterated_dist,
        timepoints=jnp.asarray(timepoints),
        rates=jnp.exp(jnp.asarray(mean_rates)),
        inv_backexchange_array=jnp.subtract(1.0, jnp.asarray(backexchange_array)),
        d2o_fraction=d2o_fraction,
        d2o_purity=d2o_purity,
        num_bins=num_bins,
    ))
    return float(np.sqrt(np.mean((exp_distributions.ravel() - thr.ravel()) ** 2)))


def diagnose_chains(
    posterior_by_chain: dict,
    undeuterated_dist: jnp.ndarray,
    timepoints: np.ndarray,
    backexchange_array: np.ndarray,
    d2o_fraction: float,
    d2o_purity: float,
    num_bins: int,
    exp_distributions: np.ndarray,
    rmse_tolerance: float = 1e-2,
) -> tuple[ChainDiagnostics, bool]:
    """
    Compute RMSE per chain. Discard chains where RMSE > min_chain_rmse + tolerance.
    Returns (ChainDiagnostics, rerun_needed).
    Preserves v1 diagnose_posterior_sample_rate_among_chains logic.
    """
    num_chains = len(posterior_by_chain['rate'])
    chain_rmse = []
    for i in range(num_chains):
        rmse = _compute_chain_rmse(
            chain_rate_samples=np.array(posterior_by_chain['rate'][i]),
            undeuterated_dist=undeuterated_dist,
            timepoints=timepoints,
            backexchange_array=backexchange_array,
            d2o_fraction=d2o_fraction,
            d2o_purity=d2o_purity,
            num_bins=num_bins,
            exp_distributions=exp_distributions,
        )
        chain_rmse.append(rmse)

    min_rmse = min(chain_rmse)
    chain_pass = [abs(r - min_rmse) <= rmse_tolerance for r in chain_rmse]
    discard = [i for i, p in enumerate(chain_pass) if not p]

    # Overall RMSE using combined summary
    summary = numpyro.diagnostics.summary(samples=posterior_by_chain)
    overall_thr = np.array(theoretical_dist_all_timepoints(
        undeuterated_dist=undeuterated_dist,
        timepoints=jnp.asarray(timepoints),
        rates=jnp.exp(jnp.asarray(summary['rate']['mean'])),
        inv_backexchange_array=jnp.subtract(1.0, jnp.asarray(backexchange_array)),
        d2o_fraction=d2o_fraction,
        d2o_purity=d2o_purity,
        num_bins=num_bins,
    ))
    overall_rmse = float(np.sqrt(np.mean((exp_distributions.ravel() - overall_thr.ravel()) ** 2)))

    diag = ChainDiagnostics(
        chain_rmse=chain_rmse,
        chain_pass=chain_pass,
        discarded_indices=discard,
        n_reruns=0,
        overall_rmse=overall_rmse,
    )
    rerun = len(discard) == num_chains
    return diag, rerun


def _discard_chains(posterior_by_chain: dict, discard_indices: list[int]) -> dict:
    """Remove chains at discard_indices from posterior dict."""
    n = len(list(posterior_by_chain.values())[0])
    keep = np.array([i for i in range(n) if i not in discard_indices])
    return {k: np.array(v)[keep] for k, v in posterior_by_chain.items()}


def _build_model_inputs(
    protein_group: ProteinGroup,
    backexchange_by_pH: dict[float, BackExchangeResult],
    undeuterated_dist: np.ndarray,
    num_bins: int,
    reference_pH: float,
) -> tuple[object, dict, np.ndarray, np.ndarray]:
    """
    Prepare model function and kwargs for run_mcmc.
    Returns (model_fn, model_kwargs, effective_timepoints, exp_distributions_stacked).
    """
    num_rates = _num_exchange_rates(protein_group.sequence)
    und = jnp.asarray(undeuterated_dist)
    pHs = protein_group.pHs

    if len(pHs) == 1:
        pH = pHs[0]
        tps = protein_group.timepoints_by_pH[pH]
        dists = protein_group.distributions_by_pH[pH]
        bx_arr = backexchange_by_pH[pH].backexchange_array

        flat_obs, nonzero_idx = _flat_nonzero(dists)

        kwargs = dict(
            num_rates=num_rates,
            undeuterated_dist=und,
            timepoints=jnp.asarray(tps),
            backexchange_array=jnp.asarray(bx_arr),
            d2o_fraction=protein_group.d2o_fraction,
            d2o_purity=protein_group.d2o_purity,
            num_bins=num_bins,
            obs_nonzero_flat=jnp.asarray(flat_obs),
            nonzero_indices=nonzero_idx,
        )
        return hdx_rate_model, kwargs, tps, bx_arr, dists

    else:
        # Multi-pH: reference pH first, remaining pHs get one merge factor each.
        # Ordering: [reference_pH, *other_pHs_sorted]
        other_pHs = sorted(pH for pH in pHs if pH != reference_pH)
        pHs_ordered = [reference_pH] + other_pHs

        tps_list = [jnp.asarray(protein_group.timepoints_by_pH[pH]) for pH in pHs_ordered]
        dists_list = [protein_group.distributions_by_pH[pH] for pH in pHs_ordered]
        bx_arrays = [backexchange_by_pH[pH].backexchange_array for pH in pHs_ordered]

        all_dists = np.vstack(dists_list)
        all_bx = np.concatenate(bx_arrays)
        flat_obs, nonzero_idx = _flat_nonzero(all_dists)
        all_tps = np.concatenate([protein_group.timepoints_by_pH[pH] for pH in pHs_ordered])

        # Auto-derive merge factor priors: |pH_i - pH_ref| log10 units apart
        merge_fac_locs = jnp.array([abs(pH - reference_pH) for pH in other_pHs])
        merge_fac_signs = jnp.array([1.0 if pH > reference_pH else -1.0
                                      for pH in other_pHs])

        kwargs = dict(
            num_rates=num_rates,
            undeuterated_dist=und,
            timepoints_list=tps_list,
            num_merge_facs=len(other_pHs),
            backexchange_array=jnp.asarray(all_bx),
            d2o_fraction=protein_group.d2o_fraction,
            d2o_purity=protein_group.d2o_purity,
            num_bins=num_bins,
            obs_nonzero_flat=jnp.asarray(flat_obs),
            nonzero_indices=nonzero_idx,
            merge_fac_locs=merge_fac_locs,
            merge_fac_signs=merge_fac_signs,
        )
        return hdx_rate_model_with_merge, kwargs, all_tps, all_bx, all_dists


def fit_rates(
    protein_group: ProteinGroup,
    backexchange_by_pH: dict[float, BackExchangeResult],
    config: MCMCConfig,
    undeuterated_dist: np.ndarray,
    num_bins: int,
    reference_pH: float | None = None,
) -> tuple[PosteriorRates, PosteriorMergeFac | None, ChainDiagnostics, np.ndarray, np.ndarray]:
    """
    Run MCMC for one protein. Handles both single-pH and dual-pH.
    Implements the rerun loop and chain discard logic from v1 BayesRateFit.fit_rate().

    Returns (PosteriorRates, PosteriorMergeFac|None, ChainDiagnostics,
             predicted_dists (T, B), rmse_per_timepoint (T,))
    """
    if reference_pH is None:
        reference_pH = min(protein_group.pHs, key=lambda pH: abs(pH - 6.0))

    numpyro.set_host_device_count(config.num_chains)
    model_fn, model_kwargs, diag_tps, diag_bx, exp_dists = _build_model_inputs(
        protein_group, backexchange_by_pH, undeuterated_dist, num_bins, reference_pH
    )
    und = jnp.asarray(undeuterated_dist)

    nuts_kernel = NUTS(model=model_fn)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        chain_method='parallel',
    )
    rng_key = random.PRNGKey(config.rng_seed)

    n_reruns = 0
    rerun = True
    posterior = None
    diagnostics = None

    while rerun:
        rng_key, run_key = random.split(rng_key)
        mcmc.run(rng_key=run_key, extra_fields=('potential_energy',), **model_kwargs)

        # For merge model, use effective timepoints for diagnostics
        posterior = sort_posterior_rates(mcmc.get_samples(group_by_chain=True))
        summary = numpyro.diagnostics.summary(samples=posterior)

        if 'merge_fac' in summary:
            mf_mean = np.atleast_1d(summary['merge_fac']['mean'])
            eff_tps = np.array(recalc_timepoints_with_merge(
                model_kwargs['timepoints_list'],
                jnp.asarray(mf_mean),
                model_kwargs['merge_fac_signs'],
            ))
        else:
            eff_tps = diag_tps

        diagnostics, rerun = diagnose_chains(
            posterior_by_chain=posterior,
            undeuterated_dist=und,
            timepoints=eff_tps,
            backexchange_array=diag_bx,
            d2o_fraction=protein_group.d2o_fraction,
            d2o_purity=protein_group.d2o_purity,
            num_bins=num_bins,
            exp_distributions=exp_dists,
            rmse_tolerance=config.rmse_tolerance,
        )
        diagnostics.n_reruns = n_reruns
        n_reruns += 1
        if rerun and n_reruns >= config.max_reruns:
            raise RuntimeError(
                f"MCMC failed to converge after {config.max_reruns} attempts "
                f"(all chains discarded). Last overall RMSE: {diagnostics.overall_rmse:.4f}"
            )

    # Discard bad chains and recompute summary
    if diagnostics.discarded_indices:
        posterior = _discard_chains(posterior, diagnostics.discarded_indices)

    summary = numpyro.diagnostics.summary(samples=posterior)

    # Build PosteriorRates — combine all chains
    all_rate_samples = np.array(posterior['rate']).reshape(-1, posterior['rate'].shape[-1])
    rate_summary = summary['rate']

    posterior_rates = PosteriorRates(
        samples=all_rate_samples,
        mean=np.atleast_1d(rate_summary['mean']),
        median=np.atleast_1d(rate_summary['median']),
        std=np.atleast_1d(rate_summary['std']),
        ci_5=np.atleast_1d(rate_summary['5.0%']),
        ci_95=np.atleast_1d(rate_summary['95.0%']),
        n_eff=np.atleast_1d(rate_summary['n_eff']),
        r_hat=np.atleast_1d(rate_summary['r_hat']),
    )

    # Build PosteriorMergeFac if dual-pH
    merge_factor = None
    if 'merge_fac' in summary:
        mf_samples = np.array(posterior['merge_fac']).reshape(-1, posterior['merge_fac'].shape[-1])
        mf_sum = summary['merge_fac']
        merge_factor = PosteriorMergeFac(
            samples=mf_samples,
            mean=np.atleast_1d(mf_sum['mean']),
            std=np.atleast_1d(mf_sum['std']),
            ci_5=np.atleast_1d(mf_sum['5.0%']),
            ci_95=np.atleast_1d(mf_sum['95.0%']),
        )
        mf_mean = merge_factor.mean
        eff_tps = np.array(recalc_timepoints_with_merge(
            model_kwargs['timepoints_list'],
            jnp.asarray(mf_mean),
            model_kwargs['merge_fac_signs'],
        ))
    else:
        eff_tps = diag_tps

    # Compute predicted distributions using posterior mean rates
    pred_dists = np.array(theoretical_dist_all_timepoints(
        undeuterated_dist=und,
        timepoints=jnp.asarray(eff_tps),
        rates=jnp.exp(jnp.asarray(posterior_rates.mean)),
        inv_backexchange_array=jnp.subtract(1.0, jnp.asarray(diag_bx)),
        d2o_fraction=protein_group.d2o_fraction,
        d2o_purity=protein_group.d2o_purity,
        num_bins=num_bins,
    ))

    rmse_per_tp = np.array([
        float(np.sqrt(np.mean((exp_dists[i] - pred_dists[i]) ** 2)))
        for i in range(len(eff_tps))
    ])

    return posterior_rates, merge_factor, diagnostics, pred_dists, rmse_per_tp
