"""
Disk I/O for the hdxrate pipeline.

read_protein_csv  — one CSV file per protein → list[InputRow]
write_results_csv — dict[protein_id, HDXResult] → summary CSV
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np

from .types import InputRow, HDXResult


def read_protein_csv(
    path: str | Path,
    d2o_fraction: float = 0.95,
    d2o_purity: float = 0.95,
    temperature: float = 295.0,
) -> list[InputRow]:
    """
    Parse one protein CSV → list[InputRow].

    Expected columns: sequence, pH, timepoint, charge, mz, intensities
    - protein_id is inferred from the filename stem
    - mz and intensities are space-separated float strings
    - d2o_fraction, d2o_purity, temperature are taken from arguments (experiment-level)
    """
    path = Path(path)
    protein_id = path.stem
    rows: list[InputRow] = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                mz = np.fromstring(row['mz'], sep=' ')
                intensities = np.fromstring(row['intensities'], sep=' ')
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:row {i+2} — failed to parse mz/intensities: {exc}") from exc

            if len(mz) == 0 or len(intensities) == 0:
                raise ValueError(f"{path}:row {i+2} — empty mz or intensities array")
            if len(mz) != len(intensities):
                raise ValueError(
                    f"{path}:row {i+2} — mz length {len(mz)} != intensities length {len(intensities)}"
                )

            rows.append(InputRow(
                protein_id=protein_id,
                sequence=row['sequence'].strip(),
                timepoint=float(row['timepoint']),
                mz=mz,
                intensities=intensities,
                charge=int(row['charge']),
                pH=float(row['pH']),
                d2o_fraction=d2o_fraction,
                d2o_purity=d2o_purity,
                temperature=temperature,
            ))

    if not rows:
        raise ValueError(f"{path} — CSV contains no data rows")

    return rows


def _arr_str(a: np.ndarray) -> str:
    """Format a 1-D array as a space-separated string of 6-significant-figure floats."""
    return ' '.join(f'{v:.6g}' for v in a)


def write_results_csv(results: dict[str, HDXResult], path: str | Path) -> None:
    """
    Write one summary row per protein to a CSV.

    Scalar columns: protein_id, sequence, pHs, rmse_total, n_eff_min, r_hat_max,
        n_reruns, backexchange_{pH} per pH, merge_factor_{mean,ci5,ci95}_{pH}
        per non-reference pH.

    Array columns (space-separated, re-parseable with np.fromstring(s, sep=' ')):
        log_rate_mean, log_rate_ci5, log_rate_ci95,
        dg_open_mean, dg_open_ci5, dg_open_ci95
    """
    path = Path(path)

    all_pHs: list[float] = sorted({pH for r in results.values() for pH in r.pHs})

    # Reference pH per protein — stored directly on HDXResult (set during inference)
    ref_pH_map = {pid: r.reference_pH for pid, r in results.items()}

    # All non-reference pHs across the dataset (for column generation)
    all_other_pHs: list[float] = sorted({
        pH
        for pid, r in results.items()
        for pH in r.pHs
        if pH != r.reference_pH
    })

    fieldnames = [
        'protein_id', 'sequence', 'pHs',
        'rmse_total', 'n_eff_min', 'r_hat_max', 'n_reruns',
    ]
    for pH in all_pHs:
        fieldnames.append(f'backexchange_{pH}')
    for pH in all_other_pHs:
        fieldnames += [
            f'merge_factor_mean_{pH}',
            f'merge_factor_ci5_{pH}',
            f'merge_factor_ci95_{pH}',
        ]
    fieldnames += [
        'log_rate_mean', 'log_rate_ci5', 'log_rate_ci95',
        'dg_open_mean', 'dg_open_ci5', 'dg_open_ci95',
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for pid, r in results.items():
            ref_pH = r.reference_pH
            other_pHs = sorted(pH for pH in r.pHs if pH != ref_pH)

            row: dict = {
                'protein_id': pid,
                'sequence': r.sequence,
                'pHs': ' '.join(str(pH) for pH in sorted(r.pHs)),
                'rmse_total': f'{r.rmse_total:.6g}',
                'n_eff_min': f'{float(np.min(r.posterior_rates.n_eff)):.1f}',
                'r_hat_max': f'{float(np.max(r.posterior_rates.r_hat)):.4f}',
                'n_reruns': r.chain_diagnostics.n_reruns,
            }

            for pH in all_pHs:
                key = f'backexchange_{pH}'
                if pH in r.backexchange:
                    row[key] = f'{r.backexchange[pH].backexchange_value:.6g}'
                else:
                    row[key] = ''

            if r.merge_factor is not None:
                for i, pH in enumerate(other_pHs):
                    if i < len(r.merge_factor.mean) and pH in all_other_pHs:
                        row[f'merge_factor_mean_{pH}'] = f'{r.merge_factor.mean[i]:.6g}'
                        row[f'merge_factor_ci5_{pH}']  = f'{r.merge_factor.ci_5[i]:.6g}'
                        row[f'merge_factor_ci95_{pH}'] = f'{r.merge_factor.ci_95[i]:.6g}'
            for pH in all_other_pHs:
                for suffix in ('mean', 'ci5', 'ci95'):
                    row.setdefault(f'merge_factor_{suffix}_{pH}', '')

            row['log_rate_mean'] = _arr_str(r.posterior_rates.mean)
            row['log_rate_ci5']  = _arr_str(r.posterior_rates.ci_5)
            row['log_rate_ci95'] = _arr_str(r.posterior_rates.ci_95)

            fe = r.free_energy
            if fe is not None and len(fe.dG_mean) > 0:
                dg_ci5  = np.percentile(fe.dG_samples, 5,  axis=0)
                dg_ci95 = np.percentile(fe.dG_samples, 95, axis=0)
                row['dg_open_mean'] = _arr_str(fe.dG_mean)
                row['dg_open_ci5']  = _arr_str(dg_ci5)
                row['dg_open_ci95'] = _arr_str(dg_ci95)
            else:
                row['dg_open_mean'] = row['dg_open_ci5'] = row['dg_open_ci95'] = ''

            writer.writerow(row)
