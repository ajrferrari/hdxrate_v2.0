from __future__ import annotations
from functools import lru_cache
import numpy as np
import molmass

from .types import InputRow, ProteinGroup

PROTON_MASS = 1.007276


def mz_to_mass_dist(
    mz: np.ndarray,
    intensities: np.ndarray,
    charge: int,
    num_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert m/z values + intensities to a 1-Da-binned, max-normalized mass distribution.

    Returns (mass_grid, dist) where mass_grid is integer masses and dist shape (num_bins,).
    If num_bins is None, uses the span of the observed masses.
    """
    masses = mz * charge - charge * PROTON_MASS
    min_mass = int(np.round(masses.min()))
    max_mass = int(np.round(masses.max()))

    span = max_mass - min_mass + 1
    n = num_bins if num_bins is not None else span

    dist = np.zeros(n)
    for m, intensity in zip(masses, intensities):
        idx = int(np.round(m)) - min_mass
        if 0 <= idx < n:
            dist[idx] += intensity

    if dist.max() > 0:
        dist = dist / dist.max()

    mass_grid = np.arange(min_mass, min_mass + n)
    return mass_grid, dist


@lru_cache(maxsize=512)
def theoretical_isotope_dist(sequence: str, num_isotopes: int) -> np.ndarray:
    """
    Compute undeuterated isotope distribution from amino acid sequence using molmass.
    Returns max-normalized array of shape (num_isotopes,).
    """
    formula = molmass.Formula(sequence)
    raw = np.array([x.intensity for x in formula.spectrum().values()])
    raw = raw / raw.max()

    if num_isotopes < len(raw):
        result = raw[:num_isotopes]
    else:
        pad = np.zeros(num_isotopes - len(raw))
        result = np.concatenate([raw, pad])

    return result.astype(np.float32)


def group_rows_to_protein_group(rows: list[InputRow]) -> ProteinGroup:
    """
    Convert a list of InputRows (all for one protein_id) into a ProteinGroup.

    Steps:
    1. Verify consistent protein_id and sequence
    2. Per row: convert (mz, intensities, charge) → mass distribution
    3. Group by pH, sort by timepoint
    4. Harmonize bin counts across all distributions
    """
    if not rows:
        raise ValueError("rows must not be empty")

    protein_id = rows[0].protein_id
    sequence = rows[0].sequence
    d2o_fraction = rows[0].d2o_fraction
    d2o_purity = rows[0].d2o_purity
    temperature = rows[0].temperature

    for r in rows:
        if r.protein_id != protein_id:
            raise ValueError(f"Mixed protein_id in rows: {r.protein_id} vs {protein_id}")
        if r.sequence != sequence:
            raise ValueError(f"Mixed sequence for protein {protein_id}")
        if r.d2o_fraction != d2o_fraction:
            raise ValueError(f"Mixed d2o_fraction for protein {protein_id}: {r.d2o_fraction} vs {d2o_fraction}")
        if r.d2o_purity != d2o_purity:
            raise ValueError(f"Mixed d2o_purity for protein {protein_id}: {r.d2o_purity} vs {d2o_purity}")
        if r.temperature != temperature:
            raise ValueError(f"Mixed temperature for protein {protein_id}: {r.temperature} vs {temperature}")

    # Convert each row → (mass_grid, dist); track all mass grids to unify
    row_data: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    all_min_masses: list[int] = []
    all_max_masses: list[int] = []

    for r in rows:
        masses = r.mz * r.charge - r.charge * PROTON_MASS
        all_min_masses.append(int(np.round(masses.min())))
        all_max_masses.append(int(np.round(masses.max())))
        row_data.append((r.timepoint, r.pH, r.mz, r.intensities, r.charge))

    global_min = min(all_min_masses)
    global_max = max(all_max_masses)
    num_bins = global_max - global_min + 1

    # Build per-pH dicts
    ph_timepoints: dict[float, list[float]] = {}
    ph_dists: dict[float, list[np.ndarray]] = {}

    for timepoint, pH, mz, intensities, charge in row_data:
        masses = mz * charge - charge * PROTON_MASS
        dist = np.zeros(num_bins)
        for m, intensity in zip(masses, intensities):
            idx = int(np.round(m)) - global_min
            if 0 <= idx < num_bins:
                dist[idx] += intensity
        if dist.max() > 0:
            dist = dist / dist.max()

        ph_timepoints.setdefault(pH, []).append(timepoint)
        ph_dists.setdefault(pH, []).append(dist)

    timepoints_by_pH: dict[float, np.ndarray] = {}
    distributions_by_pH: dict[float, np.ndarray] = {}

    for pH in ph_timepoints:
        tps = np.array(ph_timepoints[pH])
        dists = np.array(ph_dists[pH])
        order = np.argsort(tps)
        timepoints_by_pH[pH] = tps[order]
        distributions_by_pH[pH] = dists[order]  # (T, B)

    return ProteinGroup(
        protein_id=protein_id,
        sequence=sequence,
        pHs=sorted(timepoints_by_pH.keys()),
        timepoints_by_pH=timepoints_by_pH,
        distributions_by_pH=distributions_by_pH,
        d2o_fraction=d2o_fraction,
        d2o_purity=d2o_purity,
        temperature=temperature,
    )
