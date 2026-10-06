import argparse
import ast
import csv
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hdxrate.testing import generate_mock_rows
from hdxrate.pipeline import process_single_protein
from hdxrate.spectra import group_rows_to_protein_group
from hdxrate.types import MCMCConfig

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
TIMEPOINTS_FILE = DATA_DIR / "timepoints.txt"
EXAMPLES_FILE = DATA_DIR / "examples.csv"

SCENARIOS = [
    ('pH6',     [6.0],            []),
    ('pH6_9',   [6.0, 9.0],       [2.0]),
    ('pH5_6_9', [5.0, 6.0, 9.0], [1.0, 2.0]),
]


def load_timepoints():
    tps = []
    with open(TIMEPOINTS_FILE) as f:
        for line in f:
            try:
                val = float(line.strip())
                if val > 0:
                    tps.append(val)
            except ValueError:
                continue  # skip "Timepoints" header
    return np.array(sorted(tps))


def load_examples():
    proteins = []
    with open(EXAMPLES_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            proteins.append({
                'sequence': row['sequence'],
                'true_log_rates': np.array(ast.literal_eval(row['rates_mean'])),
                'true_fe': np.array(ast.literal_eval(row['free_energy'])),
            })
    return proteins


def plot_isotope_fits(pg, result, pHs, protein_id, scenario, output_dir):
    reference_pH = 6.0
    other_pHs = sorted(pH for pH in pHs if pH != reference_pH)
    pHs_ordered = [reference_pH] + other_pHs
    exp_stack = np.vstack([pg.distributions_by_pH[ph] for ph in pHs_ordered])

    labels = []
    for ph in pHs_ordered:
        for tp in pg.timepoints_by_pH[ph]:
            labels.append(f"pH{ph:.0f} {tp:.0f}s")

    n_tp = len(labels)
    max_cols = 7
    n_cols = min(n_tp, max_cols)
    n_rows = (n_tp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).ravel()

    for i in range(n_tp):
        ax = axes[i]
        exp = exp_stack[i]
        pred = result.predicted_distributions[i]
        x = np.arange(len(exp))
        ax.bar(x, exp, color='grey', alpha=0.5, width=0.8)
        ax.plot(x, pred, color='red', lw=1.5)
        rmse = result.rmse_per_timepoint[i]
        ax.set_title(f"{labels[i]} (RMSE={rmse:.3f})", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(n_tp, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{protein_id} — isotope fits ({scenario})", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "isotope_fits.png", bbox_inches='tight')
    plt.close(fig)


def plot_rates_ci(true_log_rates, result, protein_id, scenario, output_dir):
    log10 = np.log(10)
    pred_mean = result.posterior_rates.mean / log10
    pred_ci5 = result.posterior_rates.ci_5 / log10
    pred_ci95 = result.posterior_rates.ci_95 / log10
    true_log10 = true_log_rates / log10

    N = len(pred_mean)
    x = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(max(6, N * 0.3), 4))
    ax.fill_between(x, pred_ci5, pred_ci95, alpha=0.3, color='blue', label='Predicted (mean ± 90% CI)')
    ax.plot(x, pred_mean, color='blue', lw=1.5, marker='o', ms=3)
    N_true = min(len(true_log10), N)
    ax.scatter(np.arange(1, N_true + 1), true_log10[:N_true], color='red', s=20, zorder=5, label='True')
    ax.set_xlabel('Rate index (1=slowest)')
    ax.set_ylabel('log10(k) [log10 s⁻¹]')
    ax.set_title(f'{protein_id} — rates CI ({scenario})')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "rates_ci.png", bbox_inches='tight')
    plt.close(fig)


def plot_rates_scatter(true_log_rates, result, protein_id, scenario, output_dir):
    log10 = np.log(10)
    pred_mean = result.posterior_rates.mean / log10
    true_log10 = true_log_rates / log10
    n = min(len(true_log10), len(pred_mean))
    t = true_log10[:n]
    p = pred_mean[:n]

    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t) ** 2))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, s=20, alpha=0.7)
    lims = [min(t.min(), p.min()) - 0.5, max(t.max(), p.max()) + 0.5]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True log10(k)')
    ax.set_ylabel('Predicted log10(k)')
    ax.set_title(f'{protein_id} — rates scatter ({scenario})')
    ax.text(0.98, 0.05, f'MAE={mae:.3f}\nRMSE={rmse:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "rates_scatter.png", bbox_inches='tight')
    plt.close(fig)


def plot_free_energy_ci(true_fe, result, protein_id, scenario, output_dir):
    dg_samples = result.free_energy.dG_samples  # (S, N_active)

    sorted_samples = np.sort(dg_samples, axis=1)[:, ::-1]
    ci_5 = np.percentile(sorted_samples, 5, axis=0)
    ci_95 = np.percentile(sorted_samples, 95, axis=0)
    mean_sorted = np.mean(sorted_samples, axis=0)

    true_active = true_fe[true_fe > 0]
    true_active_sorted = np.sort(true_active)[::-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    x_pred = np.arange(1, len(mean_sorted) + 1)
    ax.fill_between(x_pred, ci_5, ci_95, alpha=0.3, color='blue')
    ax.plot(x_pred, mean_sorted, color='blue', lw=1.5, label='Predicted (mean ± 90% CI)')
    x_true = np.arange(1, len(true_active_sorted) + 1)
    ax.scatter(x_true, true_active_sorted, color='red', s=20, zorder=5, label='True')
    ax.set_xlabel('Residue (most → least stable)')
    ax.set_ylabel('ΔG_open (kcal/mol)')
    ax.set_title(f'{protein_id} — free energy CI ({scenario})')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "free_energy_ci.png", bbox_inches='tight')
    plt.close(fig)


def plot_free_energy_scatter(true_fe, result, protein_id, scenario, output_dir):
    true_active = true_fe[true_fe > 0]
    pred_active = result.free_energy.dG_mean
    n = min(len(true_active), len(pred_active))
    t = true_active[:n]
    p = pred_active[:n]

    mae = np.mean(np.abs(p - t))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, s=20, alpha=0.7)
    lims = [min(t.min(), p.min()) - 0.5, max(t.max(), p.max()) + 0.5]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True ΔG_open (kcal/mol)')
    ax.set_ylabel('Predicted ΔG_open (kcal/mol)')
    ax.set_title(f'{protein_id} — free energy scatter ({scenario})')
    ax.text(0.98, 0.05, f'MAE={mae:.3f} kcal/mol',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "free_energy_scatter.png", bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    if args.fast:
        mcmc_config = MCMCConfig(num_chains=2, num_warmup=100, num_samples=200)
    else:
        mcmc_config = MCMCConfig(num_chains=4, num_warmup=300, num_samples=500)

    timepoints = load_timepoints()
    proteins = load_examples()

    for i, protein in enumerate(proteins):
        for scenario_name, pHs, merge_factors in SCENARIOS:
            print(f"Running P{i}_{scenario_name}...")
            out_dir = DATA_DIR / "validation" / f"P{i}_{scenario_name}"
            out_dir.mkdir(parents=True, exist_ok=True)

            rows = generate_mock_rows(
                sequence=protein['sequence'],
                true_log_rates=protein['true_log_rates'],
                timepoints=timepoints,
                pHs=pHs,
                true_backexchange=0.0,
                noise_sigma=0.01,
                true_merge_factors=merge_factors,
                charge=3,
                rng_seed=42,
            )

            pg = group_rows_to_protein_group(rows)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='.*did not reach full deuteration.*',
                    category=RuntimeWarning,
                )
                result = process_single_protein(
                    rows,
                    mcmc_config=mcmc_config,
                    compute_free_energies=True,
                    min_fe_val=-2.0,
                )

            protein_id = f"P{i}"

            plot_isotope_fits(pg, result, pHs, protein_id, scenario_name, out_dir)
            plot_rates_ci(protein['true_log_rates'], result, protein_id, scenario_name, out_dir)
            plot_rates_scatter(protein['true_log_rates'], result, protein_id, scenario_name, out_dir)
            if result.free_energy is not None:
                plot_free_energy_ci(protein['true_fe'], result, protein_id, scenario_name, out_dir)
                plot_free_energy_scatter(protein['true_fe'], result, protein_id, scenario_name, out_dir)

            print(f"  Done. RMSE={result.rmse_total:.4f}")


if __name__ == '__main__':
    main()
