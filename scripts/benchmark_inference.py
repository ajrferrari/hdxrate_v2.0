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
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
SCENARIOS = {
    'pH6':     {'pHs': [6.0],            'merge_factors': []},
    'pH6_9':   {'pHs': [6.0, 9.0],       'merge_factors': [2.0]},
    'pH5_6_9': {'pHs': [5.0, 6.0, 9.0], 'merge_factors': [1.0, 2.0]},
}

RNG_SEED = 42


# ── Resolvable window ──────────────────────────────────────────────────────────

def resolvable_window(
    tps_all: np.ndarray,
    pHs: list[float],
    merge_factors: list[float],
) -> tuple[float, float]:
    """Union of [ln(1/t_max_eff), ln(1/t_min_eff)] across all pHs."""
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
            })
    return proteins


def load_timepoints() -> np.ndarray:
    tps = []
    with open(TIMEPOINTS_FILE) as f:
        for line in f:
            stripped = line.strip()
            try:
                val = float(stripped)
            except ValueError:
                continue  # skip header or non-numeric lines
            if val > 0:
                tps.append(val)
    return np.array(sorted(tps))


# ── Run one method ─────────────────────────────────────────────────────────────

def run_mcmc(pg, bx_by_pH, undeuterated, num_bins, cfg: MCMCConfig, dense_mass: bool = False):
    import jax.numpy as jnp
    from jax import random
    import numpyro
    from hdxrate.inference import _build_model_inputs
    from hdxrate.isotopes import theoretical_dist_all_timepoints
    from hdxrate.model import recalc_timepoints_with_merge
    from numpyro.infer import NUTS, MCMC

    ref_pH = min(pg.pHs, key=lambda pH: abs(pH - 6.0))

    numpyro.set_host_device_count(cfg.num_chains)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rates, mf, diag, pred_dists, rmse_per_tp = fit_rates(
            pg, bx_by_pH, cfg, undeuterated, num_bins, ref_pH,
        )
    wall_time = time.perf_counter() - t0

    return rates, mf, {
        'wall_time': wall_time,
        'rmse_total': float(np.mean(rmse_per_tp)),
        'guide_type': 'nuts_dense' if dense_mass else 'nuts',
    }


def run_one(protein: dict, scenario: dict, tps_all: np.ndarray,
            mcmc_cfg: MCMCConfig, svi_cfgs: dict[str, SVIConfig]) -> list[dict]:
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
    runners = [
        ('mcmc_nuts',  lambda: run_mcmc(pg, bx_by_pH, undeuterated, num_bins, mcmc_cfg, dense_mass=False)),
        ('svi_normal', lambda: fit_rates_svi(pg, bx_by_pH, svi_cfgs['normal'], undeuterated, num_bins)),
        ('svi_mvn',    lambda: fit_rates_svi(pg, bx_by_pH, svi_cfgs['mvn'],    undeuterated, num_bins)),
        ('svi_laplace',lambda: fit_rates_svi(pg, bx_by_pH, svi_cfgs['laplace'],undeuterated, num_bins)),
    ]

    for label, runner in runners:
        try:
            rates, mf, info = runner()
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
                'error': '',
            })
        except Exception as exc:
            records.append({
                'method': label,
                'mae': float('nan'),
                'coverage_90': float('nan'),
                'rmse_total': float('nan'),
                'wall_time': float('nan'),
                'window_low': window[0],
                'window_high': window[1],
                'error': str(exc),
            })
    return records


# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS = {
    'mcmc_nuts':   '#1f77b4',
    'svi_normal':  '#ff7f0e',
    'svi_mvn':     '#2ca02c',
    'svi_laplace': '#d62728',
}
METHODS = list(COLORS)


def plot_summary(all_results: dict, out_dir: Path) -> None:
    scenario_names = list(all_results)
    metrics = ['mae', 'coverage_90', 'wall_time', 'rmse_total']
    ylabels = {
        'mae': 'MAE log-rate (windowed)',
        'coverage_90': '90% CI coverage',
        'wall_time': 'Wall time (s)',
        'rmse_total': 'Fit RMSE',
    }

    for metric in metrics:
        fig, axes = plt.subplots(1, len(scenario_names),
                                  figsize=(4 * len(scenario_names), 4), sharey=False)
        if len(scenario_names) == 1:
            axes = [axes]
        for ax, scen in zip(axes, scenario_names):
            method_vals: dict[str, list[float]] = {m: [] for m in METHODS}
            for protein_records in all_results[scen].values():
                for rec in protein_records:
                    m = rec['method']
                    if m in method_vals and not np.isnan(rec[metric]):
                        method_vals[m].append(rec[metric])
            means = [np.mean(v) if v else float('nan') for v in method_vals.values()]
            stds  = [np.std(v)  if len(v) > 1 else 0.0  for v in method_vals.values()]
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
    fig, axes = plt.subplots(1, len(scenario_names),
                              figsize=(4 * len(scenario_names), 4))
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
                        help='Fewer MCMC samples and SVI steps (for testing)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proteins = load_proteins()
    tps_all = load_timepoints()

    if args.fast:
        mcmc_cfg = MCMCConfig(num_chains=2, num_warmup=50, num_samples=100,
                              rng_seed=0, max_reruns=1)
        svi_cfgs = {
            'normal':  SVIConfig(guide_type='normal',  num_steps=300, num_posterior_samples=200),
            'mvn':     SVIConfig(guide_type='mvn',     num_steps=300, num_posterior_samples=200),
            'laplace': SVIConfig(guide_type='laplace', num_steps=300, num_posterior_samples=200),
        }
    else:
        mcmc_cfg = MCMCConfig(num_chains=4, num_warmup=300, num_samples=500, rng_seed=0)
        svi_cfgs = {
            'normal':  SVIConfig(guide_type='normal',  num_steps=5000, num_posterior_samples=1000),
            'mvn':     SVIConfig(guide_type='mvn',     num_steps=5000, num_posterior_samples=1000),
            'laplace': SVIConfig(guide_type='laplace', num_steps=5000, num_posterior_samples=1000),
        }

    all_results: dict[str, dict[str, list[dict]]] = {}
    all_flat: list[dict] = []

    for scen_name, scenario in SCENARIOS.items():
        all_results[scen_name] = {}
        print(f'\n=== Scenario: {scen_name} ===')
        for i, protein in enumerate(proteins):
            pid = f'P{i}_{protein["sequence"][:8]}'
            print(f'  {pid}:', end=' ', flush=True)
            records = run_one(protein, scenario, tps_all, mcmc_cfg, svi_cfgs)
            all_results[scen_name][pid] = records
            for rec in records:
                print(f"{rec['method']}={rec['mae']:.3f}", end=' ', flush=True)
            print()
            for rec in records:
                all_flat.append({'scenario': scen_name, 'protein': pid, **rec})

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
