"""
Phase 1: Load all protein CSVs, compute the appropriate backexchange correction,
and save it to correction.pkl.

Correction type depends on the dataset:
  multi_ph  — cross-pH ODR rescue (any protein has >1 pH condition)
  single_ph — cross-protein centroid timepoint correction (all single-pH)
  none      — only one protein in the dataset; no cross-protein correction possible

Output pickle schema:
  {
    'type': 'multi_ph' | 'single_ph' | 'none',
    'data': {protein_id: {pH: BackExchangeResult}}  # multi_ph
          | TimepointCorrection                      # single_ph
          | None                                     # none
  }
"""
import pickle
from pathlib import Path

from hdxrate.io import read_protein_csv
from hdxrate.spectra import group_rows_to_protein_group
from hdxrate.backexchange import (
    fit_protein_backexchange,
    compute_timepoint_correction,
)
from hdxrate.pipeline import _cross_ph_backexchange_correction

cfg = snakemake.config  # noqa: F821 — injected by Snakemake
output_path = Path(snakemake.output.correction)  # noqa: F821
output_path.parent.mkdir(parents=True, exist_ok=True)

d2o_fraction        = cfg.get('d2o_fraction', 0.95)
d2o_purity          = cfg.get('d2o_purity',   0.95)
temperature         = cfg.get('temperature',  295.0)
min_tp_sat          = cfg.get('min_timepoints_saturation', 3)
mass_sat_threshold  = cfg.get('mass_saturation_threshold', 1.0)

# Load all protein CSVs
protein_groups = []
for csv_path in snakemake.input.csvs:  # noqa: F821
    rows = read_protein_csv(csv_path, d2o_fraction, d2o_purity, temperature)
    pg = group_rows_to_protein_group(rows)
    protein_groups.append(pg)

is_multi_ph = any(len(pg.pHs) > 1 for pg in protein_groups)

if is_multi_ph and len(protein_groups) > 1:
    bx_by_pid, obs_sat_by_pid, rescued_by_pid = _cross_ph_backexchange_correction(
        protein_groups,
        min_timepoints=min_tp_sat,
        mass_threshold=mass_sat_threshold,
    )
    # Only calibrated pHs (observed-saturated or ODR-rescued) go into precomputed dict
    calibrated_bx = {
        pg.protein_id: {
            pH: bx_by_pid[pg.protein_id][pH]
            for pH in pg.pHs
            if obs_sat_by_pid[pg.protein_id].get(pH, False)
            or pH in rescued_by_pid[pg.protein_id]
        }
        for pg in protein_groups
    }
    result = {'type': 'multi_ph', 'data': calibrated_bx}

elif len(protein_groups) > 1:
    temp_bx = {}
    for pg in protein_groups:
        pH = pg.pHs[0]
        bx = fit_protein_backexchange(
            pg.sequence,
            pg.distributions_by_pH[pH][-1],
            pg.timepoints_by_pH[pH],
            pg.d2o_fraction,
            pg.d2o_purity,
        )
        temp_bx[pg.protein_id] = bx
    tc = compute_timepoint_correction(protein_groups, temp_bx)
    result = {'type': 'single_ph', 'data': tc}

else:
    result = {'type': 'none', 'data': None}

with open(output_path, 'wb') as f:
    pickle.dump(result, f)

print(f"Correction type: {result['type']}, proteins: {[pg.protein_id for pg in protein_groups]}")
