"""
JAX-accelerated isotope distribution calculations.
All public functions are JIT-compiled. Only this module and model.py use JAX.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


@jax.jit
def poisson_binomial(success_probabilities: jnp.ndarray) -> jnp.ndarray:
    """
    Exact Poisson-binomial PMF via DFT.
    Input: (N,) success probabilities in [0, 1].
    Output: (N+1,) probability mass function.
    Preserves v1 PoiBin algorithm exactly.
    """
    n = success_probabilities.size
    omega = 2 * jnp.pi / (n + 1)

    chi = jnp.zeros(n + 1, dtype=complex)
    chi = chi.at[0].set(1)

    half_n = int(n / 2 + n % 2)
    exp_value = jnp.exp(omega * jnp.arange(1, half_n + 1) * 1j)
    xy = 1 - success_probabilities + success_probabilities * exp_value[:, jnp.newaxis]

    argz_sum = jnp.arctan2(xy.imag, xy.real).sum(axis=1)
    d_value = jnp.exp(jnp.log(jnp.abs(xy)).sum(axis=1))
    chi = chi.at[1:half_n + 1].set(d_value * jnp.exp(argz_sum * 1j))
    chi = chi.at[half_n + 1:n + 1].set(jnp.conjugate(chi[1:n - half_n + 1][::-1]))
    chi /= n + 1

    return jnp.fft.fft(chi).real


@jax.jit
def exchange_probabilities(
    timepoint: float,
    rates: jnp.ndarray,       # (N,) linear-scale rate constants (s^-1)
    inv_backexchange: float,  # 1 - backexchange_fraction
    d2o_fraction: float,
    d2o_purity: float,
) -> jnp.ndarray:
    """Per-residue HDX probability at a single timepoint."""
    return (1.0 - jnp.exp(-rates * timepoint)) * (d2o_fraction * d2o_purity * inv_backexchange)


def _theoretical_dist_single(
    undeuterated_dist: jnp.ndarray,
    timepoint: float,
    rates: jnp.ndarray,
    inv_backexchange: float,
    d2o_fraction: float,
    d2o_purity: float,
    num_bins: int,
) -> jnp.ndarray:
    probs = exchange_probabilities(timepoint, rates, inv_backexchange, d2o_fraction, d2o_purity)
    pmf = poisson_binomial(probs)
    conv = jnp.convolve(pmf, undeuterated_dist)[:num_bins]
    return conv / jnp.max(conv)


theoretical_dist_single_timepoint = jax.jit(
    _theoretical_dist_single, static_argnums=(6,)
)


def _make_vmap_body(undeuterated_dist, rates, d2o_fraction, d2o_purity, num_bins):
    def body(tp_invbx):
        return _theoretical_dist_single(
            undeuterated_dist, tp_invbx[0], rates, tp_invbx[1],
            d2o_fraction, d2o_purity, num_bins,
        )
    return body


def theoretical_dist_all_timepoints(
    undeuterated_dist: jnp.ndarray,    # (B,)
    timepoints: jnp.ndarray,           # (T,)
    rates: jnp.ndarray,                # (N,) linear-scale
    inv_backexchange_array: jnp.ndarray,  # (T,)
    d2o_fraction: float,
    d2o_purity: float,
    num_bins: int,
) -> jnp.ndarray:
    """
    Compute theoretical isotope distributions for all timepoints.
    Returns shape (T, num_bins). Vectorized via jax.vmap; JIT-compiled.
    Preserves v1 gen_theoretical_isotope_dist_for_all_timepoints pattern.
    """
    tp_invbx = jnp.stack([timepoints, inv_backexchange_array], axis=-1)
    body = _make_vmap_body(undeuterated_dist, rates, d2o_fraction, d2o_purity, num_bins)
    return jax.vmap(body)(tp_invbx)


theoretical_dist_all_timepoints = jax.jit(
    theoretical_dist_all_timepoints, static_argnums=(6,)
)
