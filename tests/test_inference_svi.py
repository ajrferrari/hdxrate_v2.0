from hdxrate.types import SVIConfig

def test_public_exports():
    import hdxrate
    assert 'SVIConfig' in hdxrate.__all__
    assert 'fit_rates_svi' in hdxrate.__all__
    assert callable(hdxrate.fit_rates_svi)
    assert isinstance(hdxrate.SVIConfig(), hdxrate.SVIConfig)


def test_sviconfig_defaults():
    cfg = SVIConfig()
    assert cfg.guide_type == 'normal'
    assert cfg.num_steps == 3000
    assert cfg.learning_rate == 0.01
    assert cfg.num_posterior_samples == 1000
    assert cfg.rng_seed == 0


import numpy as np
import pytest
from hdxrate import SVIConfig
from hdxrate.testing import generate_mock_rows
from hdxrate.inference_svi import fit_rates_svi
from hdxrate.spectra import group_rows_to_protein_group, theoretical_isotope_dist
from hdxrate.backexchange import fit_protein_backexchange

SEQUENCE = 'HMVVIPDVRGM'
# Exchangeable: positions >=2 and not P: V(2)V(3)I(4)D(6)V(7)R(8)G(9)M(10) = 8
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
