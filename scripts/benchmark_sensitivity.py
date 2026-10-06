"""
Benchmark sensitivity analysis: recovery of HDX rates and free energies
as a function of pH scenario and timepoint coverage fraction.

Usage:
    python scripts/benchmark_sensitivity.py

Outputs to data/benchmark/:
  results.pkl            — cached metrics + HDXResult for frac=1.0 runs
  mae_rates_<scenario>.png
  mae_fe_<scenario>.png
  scatter_<pid>_<scenario>.png
"""
from __future__ import annotations
import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from hdxrate import process_single_protein, MCMCConfig
from hdxrate.types import InputRow, HDXResult
from hdxrate.spectra import theoretical_isotope_dist
from hdxrate.backexchange import _gen_temp_rates, _isotope_dist_numpy

PROTON_MASS = 1.007276

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "benchmark"
OUT_DIR.mkdir(exist_ok=True)

pH_SCENARIOS: dict[str, list[float]] = {
    'pH6':     [6.0],
    'pH6_9':   [6.0, 9.0],
    'pH5_6_9': [5.0, 6.0, 9.0],
}

# Non-reference pHs for each scenario, sorted ascending.
# Order must match true_merge_factors arg in generate_mock_rows.
MERGE_FACTORS: dict[str, list[float]] = {
    'pH6':     [],           # no non-ref pHs
    'pH6_9':   [2.0],        # pH9: log10 MF=2.0 (100× faster)
    'pH5_6_9': [1.0, 2.0],   # pH5: log10 MF=1.0 (10× slower), pH9: 2.0
}

FRACTIONS = [0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
N_REPEATS_REDUCED = 5   # replicates for fractions < 1.0

MCMC_CFG = MCMCConfig(
    num_chains=2,
    num_warmup=150,
    num_samples=300,
    rng_seed=0,
    max_reruns=3,
    rmse_tolerance=0.02,
)


def load_ground_truth() -> tuple[list[dict], np.ndarray]:
    """
    Returns:
        proteins: list of dicts with keys protein_id, sequence,
                  true_log_rates (ndarray), true_free_energy (ndarray)
        tps_all:  ndarray of timepoints in seconds (excluding t=0)
    """
    df = pd.read_csv(DATA_DIR / "examples.csv")
    proteins = []
    for _, row in df.iterrows():
        proteins.append({
            'protein_id': f"P{row.name}",
            'sequence': row['sequence'],
            'true_log_rates': np.array(ast.literal_eval(row['rates_mean'])),
            'true_free_energy': np.array(ast.literal_eval(row['free_energy'])),
        })

    raw_tps = []
    for line in open(DATA_DIR / "timepoints.txt"):
        s = line.strip()
        if s and s != 'Timepoints':
            v = float(s)
            if v > 0:
                raw_tps.append(v)
    tps_all = np.array(raw_tps)
    return proteins, tps_all


def generate_rows_with_bx_noise(
    protein_id: str,
    sequence: str,
    true_log_rates: np.ndarray,
    timepoints: np.ndarray,       # shape (T,), seconds, t>0 only
    pHs: list[float],
    global_bx: float,             # global backexchange fraction (0–1)
    charge: int,
    merge_factors: list[float],   # one per non-reference pH, sorted by pH ascending
    d2o_fraction: float = 0.95,
    d2o_purity: float = 0.95,
    temperature: float = 295.0,
    bx_noise_max: float = 0.06,   # max fractional deviation: bx_tp = global_bx * (1 ± noise_max)
    noise_sigma: float = 0.01,    # Gaussian intensity noise
    num_bins: int = 30,
    rng: np.random.Generator | None = None,
) -> list[InputRow]:
    """
    Generate synthetic InputRows with per-timepoint BX noise.
    All pHs use the same global_bx (BX can't be calibrated cross-pH without a
    multi-protein dataset, per experimental design decision).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    reference_pH = min(pHs, key=lambda pH: abs(pH - 6.0))
    other_pHs = sorted(p for p in pHs if p != reference_pH)

    rates_full = _gen_temp_rates(sequence)
    exchangeable_idx = np.where(rates_full > 0)[0]
    if len(true_log_rates) != len(exchangeable_idx):
        raise ValueError(
            f"true_log_rates has {len(true_log_rates)} values but sequence has "
            f"{len(exchangeable_idx)} exchangeable residues"
        )
    rates_full[exchangeable_idx] = np.exp(true_log_rates)

    undeuterated = theoretical_isotope_dist(sequence, num_bins).astype(np.float64)
    base_mass = 500.0 + len(sequence) * 110.0
    mass_grid = np.arange(base_mass, base_mass + num_bins, dtype=float)
    mz = (mass_grid + charge * PROTON_MASS) / charge

    rows: list[InputRow] = []

    for pH in pHs:
        if pH == reference_pH:
            eff_tps = timepoints
        else:
            mf_idx = other_pHs.index(pH)
            mf = merge_factors[mf_idx]
            sign = 1.0 if pH > reference_pH else -1.0
            eff_tps = timepoints * (10.0 ** (sign * mf))

        for tp, eff_tp in zip(timepoints, eff_tps):
            noise_frac = rng.uniform(-bx_noise_max, bx_noise_max)
            tp_bx = float(np.clip(global_bx * (1.0 + noise_frac), 0.0, 1.0))
            inv_bx = 1.0 - tp_bx

            dist = _isotope_dist_numpy(
                undeuterated, eff_tp, rates_full, inv_bx,
                d2o_fraction, d2o_purity, num_bins,
            )
            noisy = dist + rng.normal(0, noise_sigma, size=num_bins)
            noisy = np.clip(noisy, 0, None)
            if noisy.max() > 0:
                noisy /= noisy.max()

            rows.append(InputRow(
                protein_id=protein_id,
                sequence=sequence,
                timepoint=float(tp),
                mz=mz.copy(),
                intensities=noisy,
                charge=charge,
                pH=float(pH),
                d2o_fraction=d2o_fraction,
                d2o_purity=d2o_purity,
                temperature=temperature,
            ))

    return rows


def sample_protein_bx(protein_idx: int, seed: int = 0) -> float:
    """Sample global backexchange in [0.20, 0.50]."""
    rng = np.random.default_rng(seed + protein_idx * 1000)
    return float(rng.uniform(0.20, 0.50))


def sample_charge(protein_idx: int, seed: int = 0) -> int:
    """Sample charge state in [5, 10]."""
    rng = np.random.default_rng(seed + protein_idx * 999)
    return int(rng.integers(5, 11))  # inclusive of 5 and 10


def compute_rate_mae(result: HDXResult, true_log_rates: np.ndarray) -> float:
    pred = result.posterior_rates.mean
    n = min(len(true_log_rates), len(pred))
    return float(np.mean(np.abs(pred[:n] - true_log_rates[:n])))


def compute_fe_mae(result: HDXResult, true_fe: np.ndarray) -> float:
    """
    MAE on residues where true free energy > 0 (structured residues only).
    Returns nan if free_energy result is None.
    """
    if result.free_energy is None:
        return float('nan')
    pred = result.free_energy.dG_mean
    mask = true_fe > 0
    n_active = mask.sum()
    n = min(n_active, len(pred))
    if n == 0:
        return float('nan')
    true_active = true_fe[mask][:n]
    pred_active = pred[:n]
    return float(np.mean(np.abs(pred_active - true_active)))


def run_single(rows: list[InputRow]) -> dict:
    """
    Run process_single_protein with no correction.
    Returns {'result': HDXResult | None, 'error': str | None}.
    """
    try:
        result = process_single_protein(
            rows=rows,
            mcmc_config=MCMC_CFG,
            backexchange_correction=None,
            precomputed_backexchange=None,
            compute_free_energies=True,
            min_fe_val=-2.0,
        )
        return {'result': result, 'error': None}
    except Exception as exc:
        return {'result': None, 'error': str(exc)}


def run_benchmark(proteins: list[dict], tps_all: np.ndarray) -> dict:
    """
    Run all (protein × scenario × fraction × replicate) combinations.

    Returns all_results dict keyed by (protein_id, scenario_name, fraction, rep_idx).
    Each value: {
        'rate_mae': float,
        'fe_mae': float,
        'rmse': float,
        'error': str | None,
        'hdx_result': HDXResult | None  # only stored for frac == 1.0
    }
    """
    cache_path = OUT_DIR / "results.pkl"
    if cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    all_results: dict = {}
    total_runs = sum(
        len(proteins) * (N_REPEATS_REDUCED if frac < 1.0 else 1)
        for frac in FRACTIONS
    ) * len(pH_SCENARIOS)
    run_idx = 0

    for prot_idx, prot in enumerate(proteins):
        pid = prot['protein_id']
        global_bx = sample_protein_bx(prot_idx)
        charge = sample_charge(prot_idx)
        print(f"\nProtein {pid}: sequence_len={len(prot['sequence'])}, "
              f"global_bx={global_bx:.3f}, charge={charge}")

        # Pre-generate full rows per scenario (once, reused by all fractions)
        full_rows_by_scenario: dict[str, list[InputRow]] = {}
        for scenario_name, pHs in pH_SCENARIOS.items():
            rng = np.random.default_rng(prot_idx * 77 + len(scenario_name))
            full_rows_by_scenario[scenario_name] = generate_rows_with_bx_noise(
                protein_id=pid,
                sequence=prot['sequence'],
                true_log_rates=prot['true_log_rates'],
                timepoints=tps_all,
                pHs=pHs,
                global_bx=global_bx,
                charge=charge,
                merge_factors=MERGE_FACTORS[scenario_name],
                rng=rng,
            )

        for scenario_name in pH_SCENARIOS:
            full_rows = full_rows_by_scenario[scenario_name]

            for frac in FRACTIONS:
                n_tp = max(4, int(round(frac * len(tps_all))))
                n_reps = 1 if frac == 1.0 else N_REPEATS_REDUCED

                for rep in range(n_reps):
                    run_idx += 1
                    key = (pid, scenario_name, frac, rep)

                    # Subsample timepoints (consistent across pHs)
                    rng_sub = np.random.default_rng(rep * 13 + int(frac * 100) * 7 + prot_idx * 500)
                    tp_indices = np.sort(
                        rng_sub.choice(len(tps_all), size=n_tp, replace=False)
                    )
                    selected_tps = set(tps_all[tp_indices].tolist())
                    sub_rows = [r for r in full_rows if r.timepoint in selected_tps]

                    out = run_single(sub_rows)
                    if out['result'] is not None:
                        r_mae = compute_rate_mae(out['result'], prot['true_log_rates'])
                        f_mae = compute_fe_mae(out['result'], prot['true_free_energy'])
                        rmse = out['result'].rmse_total
                    else:
                        r_mae = f_mae = rmse = float('nan')

                    all_results[key] = {
                        'rate_mae': r_mae,
                        'fe_mae': f_mae,
                        'rmse': rmse,
                        'error': out['error'],
                        # Store full result only at 100% (for scatter plots)
                        'hdx_result': out['result'] if frac == 1.0 else None,
                    }
                    status = f"rate_MAE={r_mae:.3f}" if not np.isnan(r_mae) else f"ERROR: {out['error'][:60]}"
                    print(f"  [{run_idx}/{total_runs}] {pid} {scenario_name} "
                          f"frac={frac:.0%} rep={rep}: {status}")

    with open(cache_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {cache_path}")
    return all_results


def plot_mae_curves(
    all_results: dict,
    proteins: list[dict],
    scenario_name: str,
    metric_key: str,    # 'rate_mae' or 'fe_mae'
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors

    for prot_idx, prot in enumerate(proteins):
        pid = prot['protein_id']
        xs, means, stds = [], [], []

        for frac in FRACTIONS:
            n_reps = 1 if frac == 1.0 else N_REPEATS_REDUCED
            vals = [
                all_results[(pid, scenario_name, frac, rep)][metric_key]
                for rep in range(n_reps)
                if (pid, scenario_name, frac, rep) in all_results
                and not np.isnan(all_results[(pid, scenario_name, frac, rep)][metric_key])
            ]
            if vals:
                xs.append(frac)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        if not xs:
            continue
        xs_arr = np.array(xs)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        c = colors[prot_idx % len(colors)]
        ax.plot(xs_arr * 100, means_arr, 'o-', color=c, label=pid, linewidth=1.5)
        ax.fill_between(
            xs_arr * 100,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.20, color=c,
        )

    ax.set_xlabel('Timepoint coverage (%)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{scenario_name} — {ylabel}')
    ax.legend(framealpha=0.7)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_mae_plots(all_results: dict, proteins: list[dict]) -> None:
    for scenario_name in pH_SCENARIOS:
        plot_mae_curves(
            all_results, proteins, scenario_name,
            'rate_mae', 'Rate MAE (log-rate units)',
            OUT_DIR / f'mae_rates_{scenario_name}.png',
        )
        plot_mae_curves(
            all_results, proteins, scenario_name,
            'fe_mae', 'Free energy MAE (kcal/mol)',
            OUT_DIR / f'mae_fe_{scenario_name}.png',
        )


def plot_scatter(
    result: HDXResult,
    true_log_rates: np.ndarray,
    true_fe: np.ndarray,
    protein_id: str,
    scenario_name: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Left panel: log-rates ──────────────────────────────────────────────
    ax = axes[0]
    pred_rates = result.posterior_rates.mean
    n = min(len(true_log_rates), len(pred_rates))
    lo_err = pred_rates[:n] - result.posterior_rates.ci_5[:n]
    hi_err = result.posterior_rates.ci_95[:n] - pred_rates[:n]
    ax.errorbar(
        true_log_rates[:n], pred_rates[:n],
        yerr=[lo_err, hi_err],
        fmt='o', ms=4, alpha=0.6, linewidth=0.5, capsize=2,
    )
    diag_min = min(true_log_rates[:n].min(), pred_rates[:n].min())
    diag_max = max(true_log_rates[:n].max(), pred_rates[:n].max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', lw=1, label='y=x')
    mae_val = float(np.mean(np.abs(pred_rates[:n] - true_log_rates[:n])))
    ax.set_xlabel('True log-rate')
    ax.set_ylabel('Predicted log-rate (posterior mean ± 90% CI)')
    ax.set_title(f'{protein_id} log-rates\nMAE={mae_val:.3f}')
    ax.legend(fontsize=8)

    # ── Right panel: free energies ─────────────────────────────────────────
    ax = axes[1]
    if result.free_energy is not None:
        pred_fe = result.free_energy.dG_mean
        mask = true_fe > 0
        n_fe = min(mask.sum(), len(pred_fe))
        true_active = true_fe[mask][:n_fe]
        pred_active = pred_fe[:n_fe]
        ax.scatter(true_active, pred_active, s=25, alpha=0.7, color='darkorange')
        diag_min = min(true_active.min(), pred_active.min())
        diag_max = max(true_active.max(), pred_active.max())
        ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', lw=1, label='y=x')
        fe_mae_val = float(np.mean(np.abs(pred_active - true_active)))
        ax.set_xlabel('True dG_open (kcal/mol)')
        ax.set_ylabel('Predicted dG_open (kcal/mol)')
        ax.set_title(f'{protein_id} free energies\nMAE={fe_mae_val:.3f}')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Free energy not available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{protein_id} free energies')

    fig.suptitle(f'{scenario_name} — 100% timepoints', fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def make_scatter_plots(all_results: dict, proteins: list[dict]) -> None:
    for prot in proteins:
        pid = prot['protein_id']
        for scenario_name in pH_SCENARIOS:
            key = (pid, scenario_name, 1.0, 0)
            if key not in all_results:
                continue
            entry = all_results[key]
            result = entry.get('hdx_result')
            if result is None:
                print(f"Skipping scatter for {pid} {scenario_name}: "
                      f"error={entry.get('error', 'unknown')}")
                continue
            plot_scatter(
                result,
                prot['true_log_rates'],
                prot['true_free_energy'],
                pid,
                scenario_name,
                OUT_DIR / f'scatter_{pid}_{scenario_name}.png',
            )


if __name__ == '__main__':
    proteins, tps_all = load_ground_truth()
    print(f"Loaded {len(proteins)} proteins, {len(tps_all)} timepoints")
    for p in proteins:
        print(f"  {p['protein_id']}: seq_len={len(p['sequence'])}, "
              f"n_rates={len(p['true_log_rates'])}, n_fe={len(p['true_free_energy'])}")

    print(f"\nOutput directory: {OUT_DIR.resolve()}")
    print(f"Total scenarios x fractions x reps = "
          f"{len(pH_SCENARIOS)} x {len(FRACTIONS)} x up to {N_REPEATS_REDUCED}")

    all_results = run_benchmark(proteins, tps_all)

    print("\nGenerating MAE curves...")
    make_mae_plots(all_results, proteins)

    print("\nGenerating scatter plots...")
    make_scatter_plots(all_results, proteins)

    # Print summary table
    print("\n=== Summary (100% timepoints) ===")
    print(f"{'Protein':<6} {'Scenario':<12} {'Rate MAE':>10} {'FE MAE':>10} {'RMSE':>10}")
    for prot in proteins:
        pid = prot['protein_id']
        for scenario_name in pH_SCENARIOS:
            key = (pid, scenario_name, 1.0, 0)
            if key in all_results:
                e = all_results[key]
                print(f"{pid:<6} {scenario_name:<12} "
                      f"{e['rate_mae']:>10.3f} {e['fe_mae']:>10.3f} {e['rmse']:>10.4f}")
    print("\nDone.")
