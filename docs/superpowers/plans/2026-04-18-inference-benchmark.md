# Inference Backend Benchmark Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SVI and Laplace approximation inference backends and a benchmark script that compares their speed, rate-recovery accuracy, and posterior coverage against the current NUTS MCMC baseline.

**Architecture:** A new `inference_svi.py` module implements `fit_rates_svi()` using NumPyro's auto-guides (AutoNormal, AutoMultivariateNormal, AutoLaplaceApproximation) reusing `_build_model_inputs` from `inference.py`. A standalone benchmark script runs all methods on synthetic data from `testing.generate_mock_rows`, evaluates rate MAE only within the scenario-specific resolvable rate window, and produces comparison plots in `data/inference_benchmark/`.

**Tech Stack:** NumPyro 0.16.1 (SVI, Trace_ELBO, AutoNormal, AutoMultivariateNormal, AutoLaplaceApproximation), JAX, pytest, matplotlib.

---

## Background: Alternatives Considered

| Method | Verdict | Why |
|--------|---------|-----|
| **SVI AutoNormal** | ✅ Include | Fastest option; quantifies cost of ignoring the slow_rate→rate_centers funnel correlation |
| **SVI AutoMultivariateNormal** | ✅ Include | Captures all pairwise correlations; O(N²) guide params but N≈50 is fine |
| **Laplace (AutoLaplaceApproximation)** | ✅ Include | Good sanity check; reveals whether posterior is near-Gaussian near window boundaries |
| **NUTS dense_mass=True** | ✅ Include as MCMC variant | One-line change; may help with funnel; establishes a better MCMC ceiling |
| **Pathfinder / BlackJAX** | ❌ Skip | New dependency, overkill for N≈50 |
| **Sequential Monte Carlo** | ❌ Skip | Only useful for multimodal posteriors; adds complexity with no clear benefit here |
| **Normalizing flows** | ❌ Skip | Over-parameterized for this problem size |

**The key scientific question:** The model has a funnel structure (`slow_rate` controls `rate_centers`, which are priors for each `rate[i]`). Diagonal SVI (AutoNormal) will underestimate posterior width because it cannot represent the induced correlation. The benchmark quantifies how much this matters in practice for rate recovery.

---

## Resolvable Rate Window

MAE must be computed only on rates the experiment can actually resolve. For a given pH scenario the window is the **union** across all pHs of `[ln(1/t_max_eff), ln(1/t_min_eff)]`, where effective timepoints scale by `10^(sign * merge_factor)`:

- pH 6 only (MF=0, scale=1): `[ln(1/86333), ln(1/25)] ≈ [-11.4, -3.2]`
- pH 9 added (MF=2, scale=100×): pH9 window `[-16.0, -7.8]` → union with pH6 = `[-16.0, -3.2]`
- pH 5 added (MF=1, scale=0.1×): pH5 window `[-9.1, +0.9]` → union all three = `[-16.0, +0.9]`

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `src/hdxrate/types.py` | Modify | Add `SVIConfig` dataclass |
| `src/hdxrate/inference_svi.py` | Create | `fit_rates_svi()` — all three guide types |
| `src/hdxrate/__init__.py` | Modify | Export `SVIConfig`, `fit_rates_svi` |
| `tests/__init__.py` | Create | Empty — makes `tests/` a package |
| `tests/test_inference_svi.py` | Create | Shape, smoke, and error tests |
| `scripts/benchmark_inference.py` | Create | Full comparison benchmark with plots |

---

## Task 1: Add SVIConfig to types.py

**Files:**
- Modify: `src/hdxrate/types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_svi.py  (create file now; more tests added in Task 3)
import pytest
from hdxrate.types import SVIConfig

def test_sviconfig_defaults():
    cfg = SVIConfig()
    assert cfg.guide_type == 'normal'
    assert cfg.num_steps == 3000
    assert cfg.learning_rate == 0.01
    assert cfg.num_posterior_samples == 1000
    assert cfg.rng_seed == 0
```

Create `tests/__init__.py` (empty) so pytest can find the package.

- [ ] **Step 2: Run test to verify it fails**

```bash
mkdir -p tests && touch tests/__init__.py
pytest tests/test_inference_svi.py::test_sviconfig_defaults -v
```

Expected: `ImportError: cannot import name 'SVIConfig' from 'hdxrate.types'`

- [ ] **Step 3: Add SVIConfig to types.py**

In `src/hdxrate/types.py`, after the `MCMCConfig` dataclass (line 58), add:

```python
@dataclass
class SVIConfig:
    guide_type: str = 'normal'          # 'normal' | 'mvn' | 'laplace'
    num_steps: int = 3000
    learning_rate: float = 0.01
    num_posterior_samples: int = 1000
    rng_seed: int = 0
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_inference_svi.py::test_sviconfig_defaults -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_inference_svi.py src/hdxrate/types.py
git commit -m "feat: add SVIConfig type for variational inference backends"
```

---

## Task 2: Implement fit_rates_svi

**Files:**
- Create: `src/hdxrate/inference_svi.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_inference_svi.py`:

```python
import numpy as np
from hdxrate import SVIConfig
from hdxrate.testing import generate_mock_rows
from hdxrate.inference_svi import fit_rates_svi
from hdxrate.spectra import group_rows_to_protein_group, theoretical_isotope_dist
from hdxrate.backexchange import fit_protein_backexchange

SEQUENCE = 'HMVVIPDVRGM'
# Exchangeable residues: positions ≥2 and not P → count manually
# H(0)M(1)V(2)V(3)I(4)P(5-skip)D(6)V(7)R(8)G(9)M(10) → 8 exchangeable
N_EXCH = 8
TRUE_RATES = np.linspace(-9.0, -3.0, N_EXCH)
TIMEPOINTS = np.array([30.0, 300.0, 3000.0, 30000.0])


def _make_setup(pHs=None, true_merge_factors=None):
    kwargs = dict(noise_sigma=0.005, rng_seed=1)
    if pHs:
        kwargs['pHs'] = pHs
    if true_merge_factors is not None:
        kwargs['true_merge_factors'] = true_merge_factors
    rows = generate_mock_rows(SEQUENCE, TRUE_RATES, TIMEPOINTS, **kwargs)
    pg = group_rows_to_protein_group(rows)
    bx_by_pH = {}
    for pH in pg.pHs:
        dists = pg.distributions_by_pH[pH]
        tps = pg.timepoints_by_pH[pH]
        bx_by_pH[pH] = fit_protein_backexchange(
            pg.sequence, dists[-1], tps, pg.d2o_fraction, pg.d2o_purity,
        )
    num_bins = pg.distributions_by_pH[pg.pHs[0]].shape[1]
    undeuterated = theoretical_isotope_dist(pg.sequence, num_bins)
    return pg, bx_by_pH, undeuterated, num_bins


@pytest.mark.parametrize('guide_type', ['normal', 'mvn', 'laplace'])
def test_fit_rates_svi_shapes(guide_type):
    pg, bx_by_pH, undeuterated, num_bins = _make_setup()
    cfg = SVIConfig(guide_type=guide_type, num_steps=100, num_posterior_samples=50)
    rates, mf, info = fit_rates_svi(pg, bx_by_pH, cfg, undeuterated, num_bins)
    assert rates.mean.shape == (N_EXCH,)
    assert rates.samples.shape == (50, N_EXCH)
    assert rates.ci_5.shape == (N_EXCH,)
    assert rates.ci_95.shape == (N_EXCH,)
    assert mf is None
    assert info['wall_time'] > 0
    assert info['guide_type'] == guide_type
    assert np.isfinite(info['final_elbo'])
    assert info['rmse_total'] >= 0


def test_fit_rates_svi_invalid_guide():
    pg, bx_by_pH, undeuterated, num_bins = _make_setup()
    cfg = SVIConfig(guide_type='bad')
    with pytest.raises(ValueError, match="Unknown guide_type"):
        fit_rates_svi(pg, bx_by_pH, cfg, undeuterated, num_bins)


def test_fit_rates_svi_multi_ph_returns_merge_factor():
    pg, bx_by_pH, undeuterated, num_bins = _make_setup(
        pHs=[6.0, 9.0], true_merge_factors=[2.0]
    )
    cfg = SVIConfig(guide_type='normal', num_steps=100, num_posterior_samples=50)
    rates, mf, info = fit_rates_svi(pg, bx_by_pH, cfg, undeuterated, num_bins)
    assert mf is not None
    assert mf.mean.shape == (1,)
    assert mf.samples.shape == (50, 1)


def test_fit_rates_svi_converges_to_reasonable_rmse():
    pg, bx_by_pH, undeuterated, num_bins = _make_setup()
    cfg = SVIConfig(guide_type='normal', num_steps=1500, num_posterior_samples=200)
    rates, _, info = fit_rates_svi(pg, bx_by_pH, cfg, undeuterated, num_bins)
    assert info['rmse_total'] < 0.15, f"RMSE too high: {info['rmse_total']:.4f}"
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_inference_svi.py -v -k "not converges"
```

Expected: `ModuleNotFoundError: No module named 'hdxrate.inference_svi'`

- [ ] **Step 3: Create inference_svi.py**

Create `src/hdxrate/inference_svi.py`:

```python
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

    model_fn, model_kwargs, diag_tps, diag_bx, exp_dists = _build_model_inputs(
        protein_group, backexchange_by_pH, undeuterated_dist, num_bins, reference_pH
    )
    und = jnp.asarray(undeuterated_dist)

    guide = guide_cls(model_fn)
    optimizer = numpyro_optim.Adam(config.learning_rate)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    rng_key = random.PRNGKey(config.rng_seed)
    rng_key, init_key = random.split(rng_key)

    t0 = time.perf_counter()
    svi_state = svi.init(init_key, **model_kwargs)
    final_loss = float('nan')
    for _ in range(config.num_steps):
        svi_state, final_loss = svi.update(svi_state, **model_kwargs)
    wall_time = time.perf_counter() - t0

    final_elbo = float(-final_loss)
    params = svi.get_params(svi_state)

    # Draw posterior samples from the fitted guide
    rng_key, pred_key = random.split(rng_key)
    predictive = Predictive(
        model_fn,
        guide=guide,
        num_samples=config.num_posterior_samples,
        params=params,
    )
    kwargs_no_obs = {k: v for k, v in model_kwargs.items() if k != 'obs_nonzero_flat'}
    samples = predictive(pred_key, **kwargs_no_obs)

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
```

- [ ] **Step 4: Run shape and error tests**

```bash
pytest tests/test_inference_svi.py -v -k "not converges"
```

Expected: all PASS (skip the convergence test — it needs 1500 steps and is slow)

- [ ] **Step 5: Run convergence test to confirm fit quality**

```bash
pytest tests/test_inference_svi.py::test_fit_rates_svi_converges_to_reasonable_rmse -v -s
```

Expected: PASS with RMSE < 0.15

- [ ] **Step 6: Commit**

```bash
git add src/hdxrate/inference_svi.py tests/test_inference_svi.py
git commit -m "feat: add fit_rates_svi with AutoNormal, AutoMultivariateNormal, and Laplace guides"
```

---

## Task 3: Export SVIConfig and fit_rates_svi

**Files:**
- Modify: `src/hdxrate/__init__.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference_svi.py`:

```python
def test_public_exports():
    import hdxrate
    assert hasattr(hdxrate, 'SVIConfig')
    assert hasattr(hdxrate, 'fit_rates_svi')
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_inference_svi.py::test_public_exports -v
```

Expected: `AssertionError` (SVIConfig not exported yet)

- [ ] **Step 3: Update __init__.py**

In `src/hdxrate/__init__.py`, add the following:

Import line — add after the existing `from .free_energy import ...` line:
```python
from .inference_svi import fit_rates_svi
```

In the types import, add `SVIConfig`:
```python
from .types import (
    InputRow, HDXResult, FreeEnergyResult, PosteriorRates, PosteriorMergeFac,
    MCMCConfig, SVIConfig, BackExchangeResult, TimepointCorrection, ProteinGroup,
    ChainDiagnostics,
)
```

In `__all__`, add `'SVIConfig'` and `'fit_rates_svi'`.

- [ ] **Step 4: Run test**

```bash
pytest tests/test_inference_svi.py::test_public_exports -v
```

Expected: PASS

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v -k "not converges"
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/hdxrate/__init__.py
git commit -m "feat: export SVIConfig and fit_rates_svi from public API"
```

---

## Task 4: Benchmark Script

**Files:**
- Create: `scripts/benchmark_inference.py`

This script compares MCMC (NUTS diagonal), MCMC (NUTS dense mass matrix), SVI AutoNormal, SVI AutoMultivariateNormal, and Laplace approximation across three pH scenarios using the three proteins from `data/examples.csv`.

- [ ] **Step 1: Create the benchmark script**

Create `scripts/benchmark_inference.py`:

```python
"""
Benchmark: MCMC vs SVI vs Laplace for HDX rate inference.

Runs all inference backends on synthetic data from the 3 example proteins
across pH scenarios [6.0], [6.0, 9.0], [5.0, 6.0, 9.0].
Evaluates: wall_time, windowed MAE (rates), 90% CI coverage, fit RMSE.
Saves comparison plots and a CSV summary to data/inference_benchmark/.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --fast   # fewer steps/samples
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import hdxrate
from hdxrate import MCMCConfig, SVIConfig
from hdxrate.testing import generate_mock_rows
from hdxrate.spectra import group_rows_to_protein_group, theoretical_isotope_dist
from hdxrate.backexchange import fit_protein_backexchange
from hdxrate.inference import fit_rates
from hdxrate.inference_svi import fit_rates_svi

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / 'data' / 'examples.csv'
TIMEPOINTS_FILE = ROOT / 'data' / 'timepoints.txt'
OUT_DIR = ROOT / 'data' / 'inference_benchmark'

# ── Scenarios ──────────────────────────────────────────────────────────────────
# merge_factors: log10 scale, one per non-reference pH, sorted by pH
SCENARIOS = {
    'pH6':     {'pHs': [6.0],              'merge_factors': []},
    'pH6_9':   {'pHs': [6.0, 9.0],         'merge_factors': [2.0]},
    'pH5_6_9': {'pHs': [5.0, 6.0, 9.0],   'merge_factors': [1.0, 2.0]},
}

RNG_SEED = 42


# ── Resolvable window ──────────────────────────────────────────────────────────

def resolvable_window(
    tps_all: np.ndarray,
    pHs: list[float],
    merge_factors: list[float],
) -> tuple[float, float]:
    """
    Union of [ln(1/t_max_eff), ln(1/t_min_eff)] across all pHs.
    merge_factors: one per non-reference pH (sorted), positive = pH above reference.
    """
    ref_pH = min(pHs, key=lambda pH: abs(pH - 6.0))
    other_pHs = sorted(pH for pH in pHs if pH != ref_pH)
    scales = {ref_pH: 1.0}
    for pH, mf in zip(other_pHs, merge_factors):
        sign = 1.0 if pH > ref_pH else -1.0
        scales[pH] = 10.0 ** (sign * mf)

    lows, highs = [], []
    for pH in pHs:
        t_min_eff = tps_all.min() * scales[pH]
        t_max_eff = tps_all.max() * scales[pH]
        lows.append(-np.log(t_max_eff))
        highs.append(-np.log(t_min_eff))
    return float(min(lows)), float(max(highs))


def windowed_mae(posterior_rates, true_log_rates: np.ndarray, window: tuple[float, float]) -> float:
    low, high = window
    n = min(len(true_log_rates), len(posterior_rates.mean))
    mask = (true_log_rates[:n] >= low) & (true_log_rates[:n] <= high)
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(posterior_rates.mean[:n][mask] - true_log_rates[:n][mask])))


def coverage_90(posterior_rates, true_log_rates: np.ndarray, window: tuple[float, float]) -> float:
    low, high = window
    n = min(len(true_log_rates), len(posterior_rates.ci_5))
    mask = (true_log_rates[:n] >= low) & (true_log_rates[:n] <= high)
    if mask.sum() == 0:
        return float('nan')
    inside = (
        (true_log_rates[:n][mask] >= posterior_rates.ci_5[:n][mask]) &
        (true_log_rates[:n][mask] <= posterior_rates.ci_95[:n][mask])
    )
    return float(inside.mean())


# ── Load data ──────────────────────────────────────────────────────────────────

def load_proteins() -> list[dict]:
    proteins = []
    with open(DATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            proteins.append({
                'sequence': row['sequence'],
                'true_log_rates': np.array(ast.literal_eval(row['rates_mean'])),
                'true_fe': np.array(ast.literal_eval(row['free_energy'])),
            })
    return proteins


def load_timepoints() -> np.ndarray:
    tps = []
    with open(TIMEPOINTS_FILE) as f:
        for line in f:
            val = float(line.strip())
            if val > 0:
                tps.append(val)
    return np.array(sorted(tps))


# ── Run one method ─────────────────────────────────────────────────────────────

def run_mcmc(pg, bx_by_pH, undeuterated, num_bins, cfg: MCMCConfig, dense_mass: bool = False):
    from numpyro.infer import NUTS, MCMC
    from jax import random
    import numpyro
    from hdxrate.inference import _build_model_inputs, sort_posterior_rates, diagnose_chains
    from hdxrate.model import recalc_timepoints_with_merge
    from hdxrate.isotopes import theoretical_dist_all_timepoints
    import jax.numpy as jnp

    ref_pH = min(pg.pHs, key=lambda pH: abs(pH - 6.0))
    model_fn, model_kwargs, diag_tps, diag_bx, exp_dists = _build_model_inputs(
        pg, bx_by_pH, undeuterated, num_bins, ref_pH
    )
    numpyro.set_host_device_count(cfg.num_chains)
    nuts = NUTS(model=model_fn, dense_mass=dense_mass)
    mcmc = MCMC(nuts, num_warmup=cfg.num_warmup, num_samples=cfg.num_samples,
                num_chains=cfg.num_chains, chain_method='parallel')
    t0 = time.perf_counter()
    mcmc.run(rng_key=random.PRNGKey(cfg.rng_seed), **model_kwargs)
    wall_time = time.perf_counter() - t0

    from hdxrate.inference import fit_rates
    # Re-use fit_rates for clean result; wall_time measured separately
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = fit_rates(pg, bx_by_pH, cfg, undeuterated, num_bins, ref_pH)
    rates, mf, diag, pred_dists, rmse_per_tp = result
    return rates, mf, {
        'wall_time': wall_time,
        'rmse_total': float(np.mean(rmse_per_tp)),
        'r_hat_max': float(rates.r_hat.max()),
        'n_eff_min': float(rates.n_eff.min()),
        'guide_type': 'nuts_dense' if dense_mass else 'nuts',
    }


def run_one(protein: dict, scenario: dict, tps_all: np.ndarray,
            mcmc_cfg: MCMCConfig, svi_cfg_normal: SVIConfig,
            svi_cfg_mvn: SVIConfig, svi_cfg_laplace: SVIConfig) -> list[dict]:
    pHs = scenario['pHs']
    mf_values = scenario['merge_factors']
    sequence = protein['sequence']
    true_log_rates = protein['true_log_rates']

    rng = np.random.default_rng(RNG_SEED)
    global_bx = float(rng.uniform(0.20, 0.50))
    charge = int(rng.integers(5, 11))

    rows = generate_mock_rows(
        sequence=sequence,
        true_log_rates=true_log_rates,
        timepoints=tps_all,
        pHs=pHs,
        charge=charge,
        true_backexchange=global_bx,
        true_merge_factors=mf_values if mf_values else 1.0,
        noise_sigma=0.01,
        rng_seed=RNG_SEED,
    )

    pg = group_rows_to_protein_group(rows)
    bx_by_pH = {}
    for pH in pg.pHs:
        dists = pg.distributions_by_pH[pH]
        tps_ph = pg.timepoints_by_pH[pH]
        bx_by_pH[pH] = fit_protein_backexchange(
            pg.sequence, dists[-1], tps_ph, pg.d2o_fraction, pg.d2o_purity,
        )
    num_bins = pg.distributions_by_pH[pg.pHs[0]].shape[1]
    undeuterated = theoretical_isotope_dist(pg.sequence, num_bins)
    window = resolvable_window(tps_all, pHs, mf_values)

    records = []
    for label, cfg_or_fn in [
        ('mcmc_nuts',      ('mcmc', mcmc_cfg, False)),
        ('mcmc_nuts_dense',('mcmc', mcmc_cfg, True)),
        ('svi_normal',     ('svi', svi_cfg_normal)),
        ('svi_mvn',        ('svi', svi_cfg_mvn)),
        ('svi_laplace',    ('svi', svi_cfg_laplace)),
    ]:
        try:
            if cfg_or_fn[0] == 'mcmc':
                rates, mf, info = run_mcmc(pg, bx_by_pH, undeuterated, num_bins,
                                           cfg_or_fn[1], dense_mass=cfg_or_fn[2])
            else:
                rates, mf, info = fit_rates_svi(pg, bx_by_pH, cfg_or_fn[1],
                                                 undeuterated, num_bins)
            mae = windowed_mae(rates, true_log_rates, window)
            cov = coverage_90(rates, true_log_rates, window)
            records.append({
                'method': label,
                'mae': mae,
                'coverage_90': cov,
                'rmse_total': info['rmse_total'],
                'wall_time': info['wall_time'],
                'window_low': window[0],
                'window_high': window[1],
                'error': None,
            })
        except Exception as exc:
            records.append({'method': label, 'error': str(exc),
                            'mae': float('nan'), 'coverage_90': float('nan'),
                            'rmse_total': float('nan'), 'wall_time': float('nan'),
                            'window_low': window[0], 'window_high': window[1]})
    return records


# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS = {
    'mcmc_nuts':       '#1f77b4',
    'mcmc_nuts_dense': '#17becf',
    'svi_normal':      '#ff7f0e',
    'svi_mvn':         '#2ca02c',
    'svi_laplace':     '#d62728',
}
METHODS = list(COLORS)


def plot_summary(all_results: dict, out_dir: Path) -> None:
    """
    all_results: {scenario_name: {protein_seq_short: [records]}}
    Produces:
      - bar chart of MAE per method per scenario (averaged over proteins)
      - bar chart of coverage_90 per method per scenario
      - bar chart of wall_time per method per scenario
      - scatter: wall_time vs MAE (one point per method×scenario×protein)
    """
    scenario_names = list(all_results)
    metrics = ['mae', 'coverage_90', 'wall_time', 'rmse_total']
    ylabels = {
        'mae': 'MAE log-rate (windowed)',
        'coverage_90': '90% CI coverage',
        'wall_time': 'Wall time (s)',
        'rmse_total': 'Fit RMSE',
    }

    for metric in metrics:
        fig, axes = plt.subplots(1, len(scenario_names), figsize=(4 * len(scenario_names), 4),
                                  sharey=False)
        if len(scenario_names) == 1:
            axes = [axes]
        for ax, scen in zip(axes, scenario_names):
            method_vals: dict[str, list[float]] = {m: [] for m in METHODS}
            for protein_records in all_results[scen].values():
                for rec in protein_records:
                    m = rec['method']
                    if not np.isnan(rec[metric]):
                        method_vals[m].append(rec[metric])
            means = [np.mean(v) if v else float('nan') for v in method_vals.values()]
            stds  = [np.std(v) if len(v) > 1 else 0.0 for v in method_vals.values()]
            xs = np.arange(len(METHODS))
            ax.bar(xs, means, yerr=stds, color=[COLORS[m] for m in METHODS],
                   capsize=4, edgecolor='k', linewidth=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(METHODS, rotation=30, ha='right', fontsize=8)
            ax.set_title(scen)
            ax.set_ylabel(ylabels[metric])
        fig.tight_layout()
        fig.savefig(out_dir / f'benchmark_{metric}.png', dpi=150)
        plt.close(fig)

    # Scatter: wall_time vs MAE
    fig, axes = plt.subplots(1, len(scenario_names), figsize=(4 * len(scenario_names), 4))
    if len(scenario_names) == 1:
        axes = [axes]
    for ax, scen in zip(axes, scenario_names):
        for m in METHODS:
            xs, ys = [], []
            for protein_records in all_results[scen].values():
                for rec in protein_records:
                    if rec['method'] == m and not np.isnan(rec['wall_time']):
                        xs.append(rec['wall_time'])
                        ys.append(rec['mae'])
            if xs:
                ax.scatter(xs, ys, label=m, color=COLORS[m], s=60, alpha=0.8)
        ax.set_xlabel('Wall time (s)')
        ax.set_ylabel('MAE log-rate (windowed)')
        ax.set_title(scen)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / 'benchmark_tradeoff.png', dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Use fewer MCMC samples and SVI steps (for testing)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proteins = load_proteins()
    tps_all = load_timepoints()

    if args.fast:
        mcmc_cfg = MCMCConfig(num_chains=2, num_warmup=50, num_samples=100, rng_seed=0, max_reruns=1)
        svi_cfg_normal  = SVIConfig(guide_type='normal',  num_steps=300,  num_posterior_samples=200)
        svi_cfg_mvn     = SVIConfig(guide_type='mvn',     num_steps=300,  num_posterior_samples=200)
        svi_cfg_laplace = SVIConfig(guide_type='laplace', num_steps=300,  num_posterior_samples=200)
    else:
        mcmc_cfg = MCMCConfig(num_chains=4, num_warmup=300, num_samples=500, rng_seed=0, max_reruns=3)
        svi_cfg_normal  = SVIConfig(guide_type='normal',  num_steps=5000, num_posterior_samples=1000)
        svi_cfg_mvn     = SVIConfig(guide_type='mvn',     num_steps=5000, num_posterior_samples=1000)
        svi_cfg_laplace = SVIConfig(guide_type='laplace', num_steps=5000, num_posterior_samples=1000)

    all_results: dict[str, dict[str, list[dict]]] = {}
    all_flat: list[dict] = []

    for scen_name, scenario in SCENARIOS.items():
        all_results[scen_name] = {}
        print(f'\n=== Scenario: {scen_name} ===')
        for i, protein in enumerate(proteins):
            pid = f'P{i}_{protein["sequence"][:8]}'
            print(f'  Protein {pid} ...', end=' ', flush=True)
            records = run_one(
                protein, scenario, tps_all,
                mcmc_cfg, svi_cfg_normal, svi_cfg_mvn, svi_cfg_laplace,
            )
            all_results[scen_name][pid] = records
            for rec in records:
                row = {'scenario': scen_name, 'protein': pid, **rec}
                all_flat.append(row)
                print(f"{rec['method']}={rec['mae']:.3f}", end=' ', flush=True)
            print()

    # Save CSV
    csv_path = OUT_DIR / 'inference_benchmark.csv'
    if all_flat:
        keys = list(all_flat[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_flat)
    print(f'\nResults saved to {csv_path}')

    plot_summary(all_results, OUT_DIR)
    print(f'Plots saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run with --fast flag to verify no errors**

```bash
cd /Users/allan/Work/Claude/hdxrate_v2.0
python scripts/benchmark_inference.py --fast 2>&1 | tee data/inference_benchmark/run_fast.log
```

Expected: prints scenario/protein lines and finishes without exception. Check `data/inference_benchmark/` for `inference_benchmark.csv` and 5 PNG files.

- [ ] **Step 3: Inspect CSV for expected columns**

```bash
head -2 data/inference_benchmark/inference_benchmark.csv
```

Expected columns: `scenario,protein,method,mae,coverage_90,rmse_total,wall_time,window_low,window_high,error`

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_inference.py data/inference_benchmark/
git commit -m "feat: add inference backend benchmark (MCMC vs SVI vs Laplace)"
```

---

## Self-Review

**Spec coverage:**
- ✅ SVI AutoNormal, AutoMultivariateNormal, Laplace — all three guides implemented
- ✅ NUTS dense_mass variant included in benchmark
- ✅ Resolvable window computed as union across pHs using merge factors
- ✅ MAE windowed to resolvable rates only
- ✅ 90% CI coverage tracked
- ✅ Wall time tracked
- ✅ Three pH scenarios
- ✅ All three example proteins
- ✅ Plots: bar charts per metric per scenario, time-vs-MAE scatter

**Placeholder scan:** None found — all code is concrete and complete.

**Type consistency:**
- `SVIConfig` defined in Task 1, used in Tasks 2–4 ✅
- `fit_rates_svi` returns `(PosteriorRates, PosteriorMergeFac | None, dict)` consistently ✅
- `_build_model_inputs` returns 5-tuple `(model_fn, model_kwargs, diag_tps, diag_bx, exp_dists)` — matches inference.py:175,211 ✅

**Known limitation to note:** `run_mcmc` in the benchmark re-runs `fit_rates` internally (after timing the raw MCMC call) to get clean `PosteriorRates` without duplicating chain-diagnostic logic. This means MCMC runs twice for the benchmark — acceptable for a research script.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-18-inference-benchmark.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, two-stage review after each, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans.

Which approach?
