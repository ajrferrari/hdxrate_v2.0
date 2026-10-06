"""
High-level API for HDX rate fitting.
Primary entry point: process_single_protein() for one protein at a time.
process_hdx_data() is a convenience wrapper for small datasets.
"""
from __future__ import annotations
from typing import Iterable
import warnings
import numpy as np

from .types import (
    InputRow, ProteinGroup, HDXResult, MCMCConfig, BackExchangeResult,
    TimepointCorrection,
)
from .spectra import group_rows_to_protein_group, theoretical_isotope_dist
from .backexchange import (
    fit_protein_backexchange, apply_timepoint_correction,
    check_saturation, compute_timepoint_correction,
    estimate_backexchange_from_high_pH,
)
from .inference import fit_rates
from .free_energy import compute_free_energy


def process_single_protein(
    rows: list[InputRow],
    mcmc_config: MCMCConfig | None = None,
    backexchange_correction: TimepointCorrection | None = None,
    precomputed_backexchange: dict[float, BackExchangeResult] | None = None,
    compute_free_energies: bool = True,
    min_fe_val: float = -2.0,
    reference_pH: float | None = None,
    min_timepoints_saturation: int = 3,
    mass_saturation_threshold: float = 1.0,
) -> HDXResult:
    """
    Process all HDX data rows for one protein end-to-end.

    Args:
        rows: All InputRow objects for a single protein_id. Must share the
              same sequence, d2o_fraction, d2o_purity, and temperature.
        mcmc_config: MCMC hyperparameters. Defaults to MCMCConfig() if None.
        backexchange_correction: Pre-computed cross-protein timepoint correction
                                 (from compute_timepoint_correction). Optional.
        precomputed_backexchange: Per-pH BackExchangeResult objects (e.g. from
                                  cross-pH ODR rescue). When provided, skips
                                  internal backexchange fitting for those pHs.
                                  May cover only a subset of pHs; uncovered pHs
                                  are subject to the saturation guard below.
        compute_free_energies: Whether to compute dG_open values.
        min_fe_val: Free energies below this threshold are excluded from dG_sorted.
        min_timepoints_saturation: Minimum number of timepoints required before a
                                   pH series can be declared saturated.
        mass_saturation_threshold: Max centroid shift (Da) between last and
                                   reference timepoint to declare saturation.

    Returns:
        HDXResult with posterior rates, diagnostics, and optional free energies.
    """
    if not rows:
        raise ValueError("rows must not be empty")

    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    # 1. Build ProteinGroup — converts m/z → mass distributions
    protein_group = group_rows_to_protein_group(rows)
    sequence = protein_group.sequence
    num_bins = protein_group.distributions_by_pH[protein_group.pHs[0]].shape[1]

    # Pre-compute undeuterated theoretical distribution once
    undeuterated_dist = theoretical_isotope_dist(sequence, num_bins)

    # 2. Protein-specific backexchange per pH
    backexchange_by_pH: dict[float, BackExchangeResult] = {}
    for pH in protein_group.pHs:
        if precomputed_backexchange is not None and pH in precomputed_backexchange:
            backexchange_by_pH[pH] = precomputed_backexchange[pH]
            continue
        dists = protein_group.distributions_by_pH[pH]
        tps = protein_group.timepoints_by_pH[pH]
        result = fit_protein_backexchange(
            sequence=sequence,
            final_tp_dist=dists[-1],
            timepoints=tps,
            d2o_fraction=protein_group.d2o_fraction,
            d2o_purity=protein_group.d2o_purity,
        )
        # 3. Apply timepoint correction if provided
        corrected_array = apply_timepoint_correction(
            backexchange_value=result.backexchange_value,
            timepoints=tps,
            correction=backexchange_correction,
        )
        backexchange_by_pH[pH] = BackExchangeResult(
            backexchange_value=result.backexchange_value,
            backexchange_array=corrected_array,
            fit_rmse=result.fit_rmse,
            theoretical_dist_at_saturation=result.theoretical_dist_at_saturation,
        )

    # Resolve reference pH once — used by saturation guard, MCMC, and free energy
    resolved_reference_pH = reference_pH if reference_pH is not None else min(
        protein_group.pHs, key=lambda pH: abs(pH - 6.0)
    )

    # 4. Saturation guard.
    # Single-pH: warn if the series did not saturate (backexchange estimate unreliable).
    # Multi-pH: uncovered pHs must each be independently saturated — no silent copying.
    if len(protein_group.pHs) == 1 and backexchange_correction is None:
        pH_only = protein_group.pHs[0]
        if precomputed_backexchange is None or pH_only not in precomputed_backexchange:
            if not check_saturation(
                protein_group.distributions_by_pH[pH_only],
                min_timepoints=min_timepoints_saturation,
                mass_rate_threshold=mass_saturation_threshold,
            ):
                warnings.warn(
                    f"Protein '{protein_group.protein_id}': pH {pH_only} series did not "
                    "reach full deuteration. Backexchange estimate may be unreliable.",
                    RuntimeWarning, stacklevel=2,
                )

    if len(protein_group.pHs) > 1 and backexchange_correction is None:
        uncovered_pHs = [
            pH for pH in protein_group.pHs
            if precomputed_backexchange is None or pH not in precomputed_backexchange
        ]
        if uncovered_pHs:
            pH_highest = max(protein_group.pHs)
            if not check_saturation(
                protein_group.distributions_by_pH[pH_highest],
                min_timepoints=min_timepoints_saturation,
                mass_rate_threshold=mass_saturation_threshold,
            ):
                raise ValueError(
                    f"Protein '{protein_group.protein_id}': highest-pH ({pH_highest}) series "
                    "did not saturate. Cannot rescue lower-pH backexchange without a saturated "
                    "reference. Provide a multi-protein dataset via process_hdx_data() for "
                    "cross-pH ODR rescue."
                )
            for pH in uncovered_pHs:
                if pH != pH_highest and not check_saturation(
                    protein_group.distributions_by_pH[pH],
                    min_timepoints=min_timepoints_saturation,
                    mass_rate_threshold=mass_saturation_threshold,
                ):
                    raise ValueError(
                        f"Protein '{protein_group.protein_id}': pH {pH} series did not "
                        "saturate and no cross-pH calibration is available for this condition. "
                        "Provide a multi-protein dataset via process_hdx_data() for cross-pH "
                        "ODR rescue, or supply precomputed_backexchange for this pH."
                    )

    # 5. MCMC rate fitting
    posterior_rates, merge_factor, chain_diagnostics, pred_dists, rmse_per_tp = fit_rates(
        protein_group=protein_group,
        backexchange_by_pH=backexchange_by_pH,
        config=mcmc_config,
        undeuterated_dist=undeuterated_dist,
        num_bins=num_bins,
        reference_pH=resolved_reference_pH,
    )

    rmse_total = float(np.mean(rmse_per_tp))

    # 6. Free energy calculation — use the same reference pH as inference
    fe_result = None
    if compute_free_energies:
        fe_result = compute_free_energy(
            posterior_rates=posterior_rates,
            sequence=sequence,
            temperature=protein_group.temperature,
            pH=resolved_reference_pH,
            d2o_fraction=protein_group.d2o_fraction,
            d2o_purity=protein_group.d2o_purity,
            min_fe_val=min_fe_val,
        )

    return HDXResult(
        protein_id=protein_group.protein_id,
        sequence=sequence,
        pHs=protein_group.pHs,
        reference_pH=resolved_reference_pH,
        posterior_rates=posterior_rates,
        merge_factor=merge_factor,
        backexchange=backexchange_by_pH,
        chain_diagnostics=chain_diagnostics,
        predicted_distributions=pred_dists,
        rmse_per_timepoint=rmse_per_tp,
        rmse_total=rmse_total,
        free_energy=fe_result,
    )


def _cross_ph_backexchange_correction(
    protein_groups: list,
    min_timepoints: int = 3,
    mass_threshold: float = 1.0,
) -> tuple[
    dict[str, dict[float, BackExchangeResult]],  # bx_by_pid
    dict[str, dict[float, bool]],                # obs_sat_by_pid (observed only, never mutated)
    dict[str, set[float]],                       # rescued_by_pid  (pHs that received ODR rescue)
]:
    """
    Compute per-protein per-pH BackExchangeResult with cross-pH ODR rescue.

    ODR training is always restricted to proteins with OBSERVED saturation at
    both pHs in a pair. Rescued series are never promoted to observed-saturated
    so they cannot contaminate training for subsequent lower-pH pairs.

    Returns three dicts:
      bx_by_pid       — final BackExchangeResult per protein per pH (some rescued)
      obs_sat_by_pid  — immutable observed saturation flags
      rescued_by_pid  — set of pHs where ODR actually changed the backexchange value
    """
    all_pHs = sorted({pH for pg in protein_groups for pH in pg.pHs}, reverse=True)

    bx_by_pid: dict[str, dict[float, BackExchangeResult]] = {}
    obs_sat_by_pid: dict[str, dict[float, bool]] = {}
    rescued_by_pid: dict[str, set[float]] = {}

    for pg in protein_groups:
        bx_by_pid[pg.protein_id] = {}
        obs_sat_by_pid[pg.protein_id] = {}
        rescued_by_pid[pg.protein_id] = set()
        for pH in pg.pHs:
            dists = pg.distributions_by_pH[pH]
            tps = pg.timepoints_by_pH[pH]
            bx = fit_protein_backexchange(
                pg.sequence, dists[-1], tps, pg.d2o_fraction, pg.d2o_purity,
            )
            bx_by_pid[pg.protein_id][pH] = bx
            obs_sat_by_pid[pg.protein_id][pH] = check_saturation(
                dists, min_timepoints=min_timepoints, mass_rate_threshold=mass_threshold,
            )

    for ph_high, ph_low in zip(all_pHs, all_pHs[1:]):
        proteins_with_both = [
            pg for pg in protein_groups
            if ph_high in pg.pHs and ph_low in pg.pHs
        ]
        if len(proteins_with_both) < 2:
            continue

        bkex_high = np.array([bx_by_pid[pg.protein_id][ph_high].backexchange_value
                               for pg in proteins_with_both])
        bkex_low  = np.array([bx_by_pid[pg.protein_id][ph_low].backexchange_value
                               for pg in proteins_with_both])
        # Always train ODR on OBSERVED saturation only — rescued series stay out
        obs_sat_high = np.array([obs_sat_by_pid[pg.protein_id][ph_high]
                                  for pg in proteins_with_both])
        obs_sat_low  = np.array([obs_sat_by_pid[pg.protein_id][ph_low]
                                  for pg in proteins_with_both])

        corrected_low = estimate_backexchange_from_high_pH(
            bkex_low, bkex_high, obs_sat_low, obs_sat_high,
        )

        for pg, new_bkex in zip(proteins_with_both, corrected_low):
            pid = pg.protein_id
            if not obs_sat_by_pid[pid][ph_low] and float(new_bkex) != bx_by_pid[pid][ph_low].backexchange_value:
                tps = pg.timepoints_by_pH[ph_low]
                old = bx_by_pid[pid][ph_low]
                bx_by_pid[pid][ph_low] = BackExchangeResult(
                    backexchange_value=float(new_bkex),
                    backexchange_array=np.full(len(tps), float(new_bkex)),
                    fit_rmse=old.fit_rmse,
                    theoretical_dist_at_saturation=old.theoretical_dist_at_saturation,
                )
                rescued_by_pid[pid].add(ph_low)
            # obs_sat_by_pid is never mutated — rescued ≠ observed

    return bx_by_pid, obs_sat_by_pid, rescued_by_pid


def process_hdx_data(
    rows: Iterable[InputRow],
    mcmc_config: MCMCConfig | None = None,
    compute_free_energies: bool = True,
    timepoint_correction: bool = True,
    min_fe_val: float = -2.0,
    skip_failures: bool = False,
    min_timepoints_saturation: int = 3,
    mass_saturation_threshold: float = 1.0,
) -> dict[str, HDXResult]:
    """
    Convenience wrapper: group rows by protein_id and call process_single_protein
    for each.

    For multi-pH datasets (any protein has >1 pH): automatically performs
    cross-pH ODR backexchange rescue. Only pHs that were either directly observed
    as saturated or successfully rescued by ODR are passed as precomputed; the
    saturation guard in process_single_protein handles any remaining uncalibrated pHs.

    For single-pH datasets: optionally computes a cross-protein timepoint
    correction (timepoint_correction=True, default).

    If skip_failures=False (default), raises an aggregated exception if any protein
    fails. If skip_failures=True, failed proteins are omitted from results and a
    RuntimeWarning is emitted for each.

    For large experiments, call process_single_protein() directly from Snakemake.
    """
    all_rows = list(rows)

    grouped: dict[str, list[InputRow]] = {}
    for row in all_rows:
        grouped.setdefault(row.protein_id, []).append(row)

    protein_groups = []
    for prows in grouped.values():
        try:
            protein_groups.append(group_rows_to_protein_group(prows))
        except Exception:
            pass

    # Map protein_id → ProteinGroup for quick lookup
    pg_by_pid = {pg.protein_id: pg for pg in protein_groups}

    is_multi_ph = any(len(pg.pHs) > 1 for pg in protein_groups)

    correction: TimepointCorrection | None = None
    # calibrated_bx[pid][pH] exists only for pHs that are observed-saturated or ODR-rescued
    calibrated_bx: dict[str, dict[float, BackExchangeResult]] = {}

    if is_multi_ph and len(protein_groups) > 1:
        bx_by_pid, obs_sat_by_pid, rescued_by_pid = _cross_ph_backexchange_correction(
            protein_groups,
            min_timepoints=min_timepoints_saturation,
            mass_threshold=mass_saturation_threshold,
        )
        # Only pass pHs that are confirmed calibrated to avoid bypassing guard
        for pg in protein_groups:
            pid = pg.protein_id
            calibrated_bx[pid] = {
                pH: bx_by_pid[pid][pH]
                for pH in pg.pHs
                if obs_sat_by_pid[pid].get(pH, False) or pH in rescued_by_pid[pid]
            }

    # Timepoint correction is independent of the cross-pH path.
    # In a mixed dataset, apply it only to single-pH proteins.
    single_ph_groups = [pg for pg in protein_groups if len(pg.pHs) == 1]
    if timepoint_correction and len(single_ph_groups) > 1:
        temp_bx: dict[str, BackExchangeResult] = {}
        for pg in single_ph_groups:
            pH = pg.pHs[0]
            dists = pg.distributions_by_pH[pH]
            tps = pg.timepoints_by_pH[pH]
            temp_bx[pg.protein_id] = fit_protein_backexchange(
                pg.sequence, dists[-1], tps, pg.d2o_fraction, pg.d2o_purity,
            )
        correction = compute_timepoint_correction(single_ph_groups, temp_bx)

    results: dict[str, HDXResult] = {}
    failures: dict[str, str] = {}
    for protein_id, prows in grouped.items():
        pg = pg_by_pid.get(protein_id)
        # Pass precomputed only when at least one pH is calibrated
        pre_bx = calibrated_bx.get(protein_id) or None
        try:
            results[protein_id] = process_single_protein(
                rows=prows,
                mcmc_config=mcmc_config,
                backexchange_correction=correction,
                precomputed_backexchange=pre_bx,
                compute_free_energies=compute_free_energies,
                min_fe_val=min_fe_val,
                min_timepoints_saturation=min_timepoints_saturation,
                mass_saturation_threshold=mass_saturation_threshold,
            )
        except (RuntimeError, ValueError) as exc:
            failures[protein_id] = str(exc)

    if failures:
        if skip_failures:
            import warnings
            for pid, msg in failures.items():
                warnings.warn(f"process_hdx_data: skipped '{pid}': {msg}", RuntimeWarning, stacklevel=2)
        else:
            lines = "\n".join(f"  {pid}: {msg}" for pid, msg in failures.items())
            raise RuntimeError(
                f"{len(failures)} protein(s) failed. "
                f"Pass skip_failures=True to skip and continue.\n{lines}"
            )

    return results
