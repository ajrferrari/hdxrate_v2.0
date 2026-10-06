"""
Free energy of opening calculation from HDX exchange rates.
All pure Python + numpy; no JAX.
"""
from __future__ import annotations
import math
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .types import PosteriorRates, FreeEnergyResult

R_KCAL = 1.987204e-3  # kcal mol^-1 K^-1


def milne_intrinsic_rates(
    sequence: str,
    temperature: float,
    pH: float,
    d2o_purity: float,
    d2o_fraction: float,
    nterm_mode: str = 'NT',
    cterm_mode: str = 'CT',
) -> np.ndarray:
    """
    Per-residue intrinsic exchange rates using the Milne model (JASMS 2018).
    Returns array of shape (len(sequence),) with 0.0 for non-exchangeable
    residues (first two and prolines).
    Preserves v1 calculate_intrinsic_exchange_rates exactly.
    """
    seq = [x for x in sequence]
    seq.insert(0, nterm_mode)
    seq.append(cterm_mode)

    ka = (10.0 ** 1.62) / 60
    kb = (10.0 ** 10.18) / 60
    kw = (10.0 ** -1.5) / 60

    R = 1.987
    TCorr1 = (1.0 / temperature - 1.0 / 278.0) / R
    TCorr2 = (1.0 / temperature - 1.0 / 293.0) / R

    AcidEa, BaseEa, SolventEa = 14000.0, 17000.0, 19000.0
    AspEa, GluEa, HisEa = 1000.0, 1083.0, 7500.0

    AspTCorr = math.exp(-TCorr1 * AspEa)
    GluTCorr = math.exp(-TCorr1 * GluEa)
    HisTCorr = math.exp(-TCorr1 * HisEa)

    pH_corr = pH + 0.4 * d2o_purity * d2o_fraction
    pKD = 15.05
    pKAsp = -math.log10(10 ** (-4.48) * AspTCorr)
    pKGlu = -math.log10(10 ** (-4.93) * GluTCorr)
    pKHis = -math.log10(10 ** (-7.42) * HisTCorr)

    def _protonated(mA, mA0, mB, mB0, pK, pH):
        denom = 10.0 ** -pK + 10.0 ** -pH
        la = math.log10(10 ** (mA - pH) / denom + 10 ** (mA0 - pK) / denom)
        lb = math.log10(10 ** (mB - pH) / denom + 10 ** (mB0 - pK) / denom)
        return la, lb

    MA = {
        'A': (0.00, 0.00), 'C': (-0.54, -0.46), 'C2': (-0.74, -0.58),
        'D0': (0.90, 0.58), 'D+': (-0.90, -0.12),
        'E0': (-0.90, 0.31), 'E+': (-0.60, -0.27),
        'F': (-0.52, -0.43), 'G': (-0.22, 0.22),
        'H0': (0.00, 0.00), 'H+': (-0.80, -0.51),
        'I': (-0.91, -0.59), 'K': (-0.56, -0.29), 'L': (-0.57, -0.13),
        'M': (-0.64, -0.28), 'N': (-0.58, -0.13), 'P': (0.00, -0.19),
        'Pc': (0.00, -0.85), 'Q': (-0.47, -0.27), 'R': (-0.59, -0.32),
        'S': (-0.44, -0.39), 'T': (-0.79, -0.47), 'V': (-0.74, -0.30),
        'W': (-0.40, -0.44), 'Y': (-0.41, -0.37), '?': (0.00, 0.00),
    }
    MB = {
        'A': (0.00, 0.00), 'C': (0.62, 0.55), 'C2': (0.55, 0.46),
        'D0': (0.10, -0.18), 'D+': (0.69, 0.60),
        'E0': (-0.11, -0.15), 'E+': (0.24, 0.39),
        'F': (-0.24, 0.06), 'G': (-0.03, 0.17),
        'H0': (-0.10, 0.14), 'H+': (0.80, 0.83),
        'I': (-0.73, -0.23), 'K': (-0.04, 0.12), 'L': (-0.58, -0.21),
        'M': (-0.01, 0.11), 'N': (0.49, 0.32), 'P': (0.00, -0.24),
        'Pc': (0.00, 0.60), 'Q': (0.06, 0.20), 'R': (0.08, 0.22),
        'S': (0.37, 0.30), 'T': (-0.07, 0.20), 'V': (-0.70, -0.14),
        'W': (-0.41, -0.11), 'Y': (-0.27, 0.05), '?': (0.00, 0.00),
    }

    # pH-dependent Asp, Glu, His corrections
    lAD, rAD = _protonated(MA['D+'][0], MA['D0'][0], MA['D+'][1], MA['D0'][1], pKAsp, pH_corr)
    lAE, rAE = _protonated(MA['E+'][0], MA['E0'][0], MA['E+'][1], MA['E0'][1], pKGlu, pH_corr)
    lAH, rAH = _protonated(MA['H+'][0], MA['H0'][0], MA['H+'][1], MA['H0'][1], pKHis, pH_corr)
    lBD, rBD = _protonated(MB['D+'][0], MB['D0'][0], MB['D+'][1], MB['D0'][1], pKAsp, pH_corr)
    lBE, rBE = _protonated(MB['E+'][0], MB['E0'][0], MB['E+'][1], MB['E0'][1], pKGlu, pH_corr)
    lBH, rBH = _protonated(MB['H+'][0], MB['H0'][0], MB['H+'][1], MB['H0'][1], pKHis, pH_corr)

    MA['D'] = (lAD, rAD); MA['E'] = (lAE, rAE); MA['H'] = (lAH, rAH)
    MB['D'] = (lBD, rBD); MB['E'] = (lBE, rBE); MB['H'] = (lBH, rBH)

    # Termini
    RhoAcidNT = -1.32
    LambdaAcidCT = math.log10(
        10 ** (0.05 - pH_corr) / (10 ** -pKGlu + 10 ** -pH_corr) +
        10 ** (0.96 - pKGlu) / (10 ** -pKGlu + 10 ** -pH_corr))
    RhoBaseNT = 1.62
    LambdaBaseCT = -1.80
    LambdaAcidNMe = math.log10(135.5 / (ka * 60))
    LambdaBaseNMe = math.log10(2970000000 / (kb * 60))

    MA['NT'] = (None, RhoAcidNT);   MA['CT'] = (LambdaAcidCT, None)
    MB['NT'] = (None, RhoBaseNT);   MB['CT'] = (LambdaBaseCT, None)
    MA['NMe'] = (LambdaAcidNMe, None); MB['NMe'] = (LambdaBaseNMe, None)
    MA['Ac'] = (None, 0.29);  MB['Ac'] = (None, -0.20)

    DIon = 10.0 ** -pH_corr
    ODIon = 10.0 ** (pH_corr - pKD)

    Fta = math.exp(-AcidEa * TCorr2)
    Ftb = math.exp(-BaseEa * TCorr2)
    Ftw = math.exp(-SolventEa * TCorr2)

    rates = [0.0]  # residue 0 (prepended terminus) → 0

    for i in range(2, len(seq) - 1):
        res = seq[i]
        if res in ('P', 'Pc'):
            rates.append(0.0)
            continue

        left = seq[i - 1]
        right = seq[i + 1]

        left_a = MA.get(left, MA['?'])
        left_b = MB.get(left, MB['?'])
        res_a = MA.get(res, MA['?'])
        res_b = MB.get(res, MB['?'])

        if right == 'CT':
            Fa = 10.0 ** (left_a[1] + res_a[0] + MA['CT'][0])
            Fb = 10.0 ** (left_b[1] + res_b[0] + MB['CT'][0])
        elif i == 2:
            Fa = 10.0 ** (MA['NT'][1] + left_a[1] + res_a[0])
            Fb = 10.0 ** (MB['NT'][1] + left_b[1] + res_b[0])
        else:
            Fa = 10.0 ** (left_a[1] + res_a[0])
            Fb = 10.0 ** (left_b[1] + res_b[0])

        kaT = Fa * ka * DIon * Fta
        kbT = Fb * kb * ODIon * Ftb
        kwT = Fb * kw * Ftw
        rates.append(kaT + kbT + kwT)

    return np.array(rates)


def regression_corrected_intrinsics(
    intrinsic_rates: np.ndarray,    # full-sequence intrinsic rates (with zeros)
    measured_log_rates: np.ndarray, # (N_active,) log-scale MCMC mean rates (sorted)
    sd_slope: float = 0.38,
) -> np.ndarray:
    """
    Regression-corrected intrinsic rates using measured rate z-scores.
    Preserves v1 IntrinsicRate.get_reg_intrinsics logic.
    Returns array of shape (N_active,) sorted slowest→fastest.
    """
    measured_rates = np.exp(measured_log_rates)
    valid = np.log(measured_rates) < -4
    measurable = measured_rates[valid]

    log_ints = np.log(intrinsic_rates[intrinsic_rates != 0])
    if len(measurable) > 2:
        log_meas = np.log(measured_rates)
        med_meas = np.median(log_meas[valid])
        std_meas = np.std(log_meas[valid])
        z_meas = (log_meas - med_meas) / std_meas
        reg = sd_slope * np.std(log_ints) * z_meas + np.median(log_ints)
        reg = np.clip(reg, np.min(log_ints), np.max(log_ints))
        return np.sort(np.exp(reg))

    return np.full(len(measured_log_rates), float(np.median(np.exp(log_ints))))


def dG_from_rates(
    measured_rates: np.ndarray,    # (..., N_active) linear-scale rates; vectorized
    intrinsic_rates: np.ndarray,   # (N_active,)
    temperature: float = 295.0,
) -> np.ndarray:
    """
    dG_open = -R*T * ln(k_op),  k_op = k_meas / (k_intrinsic - k_meas).
    Vectorized over leading sample dimensions.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        k_op = measured_rates / (intrinsic_rates - measured_rates)
        return -R_KCAL * temperature * np.log(k_op)


def net_charge_correction(
    dG: np.ndarray,
    sequence: str,
    pH: float,
) -> np.ndarray:
    """Apply empirical net charge correction: dG -= net_charge * (-0.12)."""
    net_charge = ProteinAnalysis(sequence).charge_at_pH(pH)
    return dG - (net_charge * -0.12)


def compute_free_energy(
    posterior_rates: PosteriorRates,
    sequence: str,
    temperature: float,
    pH: float,
    d2o_fraction: float,
    d2o_purity: float,
    apply_net_charge_corr: bool = True,
    min_fe_val: float = -2.0,
) -> FreeEnergyResult:
    """
    Full pipeline: intrinsic rates → regression correction → dG → net charge.
    Operates on the full posterior sample matrix for uncertainty quantification.
    """
    intr = milne_intrinsic_rates(sequence, temperature, pH, d2o_purity, d2o_fraction)
    reg_intr = regression_corrected_intrinsics(intr, posterior_rates.mean)

    # Posterior mean dG
    mean_dG = dG_from_rates(np.exp(posterior_rates.mean), reg_intr, temperature)
    if apply_net_charge_corr:
        mean_dG = net_charge_correction(mean_dG, sequence, pH)

    # Full posterior dG over all samples — shape (S, N_active)
    sample_rates = np.exp(posterior_rates.samples)  # (S, N_active)
    dG_samples = dG_from_rates(sample_rates, reg_intr, temperature)
    if apply_net_charge_corr:
        net_charge = ProteinAnalysis(sequence).charge_at_pH(pH)
        dG_samples = dG_samples - (net_charge * -0.12)

    # Sorted dG from mean (most → least stable, excluding below min threshold)
    dG_sorted_tmp = mean_dG.copy()
    dG_sorted_tmp[dG_sorted_tmp <= min_fe_val] = np.nan
    valid = ~np.isnan(dG_sorted_tmp)
    dG_sorted = np.sort(dG_sorted_tmp[valid])[::-1]

    return FreeEnergyResult(
        intrinsic_rates=intr,
        reg_intrinsic_rates=reg_intr,
        net_charge=float(ProteinAnalysis(sequence).charge_at_pH(pH)),
        dG_samples=dG_samples,
        dG_mean=mean_dG,
        dG_sorted=dG_sorted,
    )
