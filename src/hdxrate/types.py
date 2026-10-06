from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class InputRow:
    protein_id: str
    sequence: str
    timepoint: float          # seconds of HDX exposure
    mz: np.ndarray            # m/z values from mass spectrum
    intensities: np.ndarray   # corresponding intensities
    charge: int               # charge state
    pH: float                 # experiment pH (typically 6.0 or 9.0)
    d2o_fraction: float = 0.95
    d2o_purity: float = 0.95
    temperature: float = 295.0  # Kelvin


@dataclass
class ProteinGroup:
    """All data for one protein, grouped and converted to mass distributions."""
    protein_id: str
    sequence: str
    pHs: list[float]
    timepoints_by_pH: dict[float, np.ndarray]      # pH → sorted timepoints (T,)
    distributions_by_pH: dict[float, np.ndarray]   # pH → (T, B) normalized mass dists
    d2o_fraction: float
    d2o_purity: float
    temperature: float


@dataclass
class BackExchangeResult:
    backexchange_value: float           # scalar optimized backexchange fraction
    backexchange_array: np.ndarray      # per-timepoint backexchange, shape (T,)
    fit_rmse: float
    theoretical_dist_at_saturation: np.ndarray  # shape (B,)


@dataclass
class TimepointCorrection:
    """Cross-protein centroid-shift corrections keyed by timepoint value."""
    timepoints: np.ndarray        # shape (T,)
    correction_values: np.ndarray # shape (T,), median mass-diff rate per timepoint


@dataclass
class MCMCConfig:
    num_chains: int = 4
    num_warmup: int = 100
    num_samples: int = 250
    rmse_tolerance: float = 1e-2
    rng_seed: int = 0
    max_reruns: int = 5


@dataclass
class SVIConfig:
    guide_type: str = 'normal'          # 'normal' | 'mvn' | 'laplace'
    num_steps: int = 3000
    learning_rate: float = 0.01
    num_posterior_samples: int = 1000
    rng_seed: int = 0


@dataclass
class PosteriorRates:
    samples: np.ndarray   # (total_samples, N) — sorted slowest→fastest
    mean: np.ndarray      # (N,)
    median: np.ndarray
    std: np.ndarray
    ci_5: np.ndarray
    ci_95: np.ndarray
    n_eff: np.ndarray
    r_hat: np.ndarray


@dataclass
class PosteriorMergeFac:
    samples: np.ndarray   # (total_samples, num_merge_facs)
    mean: np.ndarray
    std: np.ndarray
    ci_5: np.ndarray
    ci_95: np.ndarray


@dataclass
class ChainDiagnostics:
    chain_rmse: list[float]
    chain_pass: list[bool]
    discarded_indices: list[int]
    n_reruns: int
    overall_rmse: float


@dataclass
class FreeEnergyResult:
    intrinsic_rates: np.ndarray      # (N,) per-residue intrinsic rates
    reg_intrinsic_rates: np.ndarray  # (N,) regression-corrected intrinsic rates
    net_charge: float
    dG_samples: np.ndarray           # (total_samples, N_active) full posterior dG
    dG_mean: np.ndarray              # (N_active,)
    dG_sorted: np.ndarray            # (N_active,) sorted most→least stable


@dataclass
class HDXResult:
    protein_id: str
    sequence: str
    pHs: list[float]
    reference_pH: float
    posterior_rates: PosteriorRates
    merge_factor: Optional[PosteriorMergeFac]
    backexchange: dict[float, BackExchangeResult]  # pH → BackExchangeResult
    chain_diagnostics: ChainDiagnostics
    predicted_distributions: np.ndarray  # (T_total, B)
    rmse_per_timepoint: np.ndarray
    rmse_total: float
    free_energy: Optional[FreeEnergyResult] = None
