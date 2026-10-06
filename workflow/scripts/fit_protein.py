"""
Phase 2: Fit one protein using its pre-computed correction.

Reads {protein}.csv + correction.pkl, calls process_single_protein,
saves HDXResult to {protein}.pkl.
"""
import pickle
from pathlib import Path

from hdxrate import MCMCConfig, process_single_protein
from hdxrate.io import read_protein_csv

cfg = snakemake.config  # noqa: F821
csv_path        = Path(snakemake.input.csv)         # noqa: F821
correction_path = Path(snakemake.input.correction)  # noqa: F821
output_path     = Path(snakemake.output.result)     # noqa: F821
output_path.parent.mkdir(parents=True, exist_ok=True)

rows = read_protein_csv(
    csv_path,
    d2o_fraction=cfg.get('d2o_fraction', 0.95),
    d2o_purity=cfg.get('d2o_purity',     0.95),
    temperature=cfg.get('temperature',   295.0),
)

mcmc_config = MCMCConfig(
    num_chains   = cfg.get('num_chains',      4),
    num_warmup   = cfg.get('num_warmup',    500),
    num_samples  = cfg.get('num_samples',  1000),
    rng_seed     = cfg.get('rng_seed',       42),
    max_reruns   = cfg.get('max_reruns',      3),
    rmse_tolerance = cfg.get('rmse_tolerance', 0.01),
)

reference_pH           = cfg.get('reference_pH') or None  # null in YAML → None
min_tp_sat             = cfg.get('min_timepoints_saturation', 3)
mass_sat_threshold     = cfg.get('mass_saturation_threshold', 1.0)

with open(correction_path, 'rb') as f:
    correction = pickle.load(f)

protein_id = csv_path.stem
correction_type = correction['type']
correction_data = correction['data']

shared_kwargs = dict(
    mcmc_config=mcmc_config,
    compute_free_energies=cfg.get('compute_free_energies', True),
    min_fe_val=cfg.get('min_fe_val', -2.0),
    reference_pH=reference_pH,
    min_timepoints_saturation=min_tp_sat,
    mass_saturation_threshold=mass_sat_threshold,
)

if correction_type == 'multi_ph':
    pre_bx = correction_data.get(protein_id) or None
    result = process_single_protein(rows=rows, precomputed_backexchange=pre_bx, **shared_kwargs)
elif correction_type == 'single_ph':
    result = process_single_protein(rows=rows, backexchange_correction=correction_data, **shared_kwargs)
else:
    result = process_single_protein(rows=rows, **shared_kwargs)

with open(output_path, 'wb') as f:
    pickle.dump(result, f)

print(f"{protein_id}: RMSE={result.rmse_total:.4f}  pHs={result.pHs}")
