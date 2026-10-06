#!/usr/bin/env python3
"""Simulate pH-dependent HDX experiments and fit rates via MCMC."""

from __future__ import annotations
import ast
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hdxrate.free_energy import milne_intrinsic_rates
from hdxrate.inference import fit_rates
from hdxrate.isotopes import theoretical_dist_all_timepoints
from hdxrate.spectra import theoretical_isotope_dist
from hdxrate.types import BackExchangeResult, MCMCConfig, ProteinGroup

# ── Constants ────────────────────────────────────────────────────────────────
R_KCAL       = 1.987e-3
TEMPERATURE  = 295.0
D2O_FRACTION = 0.95
D2O_PURITY   = 0.95
NUM_BINS     = 60
PHS          = [5.0, 6.0, 9.0]
PH_COLORS    = {5.0: "royalblue", 6.0: "darkorange", 9.0: "mediumseagreen"}
PH_LABELS    = {5.0: "pH 5", 6.0: "pH 6", 9.0: "pH 9"}

ROOT    = Path(__file__).resolve().parent.parent
DATADIR = ROOT / "data"
FIGDIR  = ROOT / "figures"


def load_data() -> list[dict]:
    """Return list of dicts with keys: protein_id, sequence, free_energies."""
    df = pd.read_csv(DATADIR / "examples.csv", index_col=0)
    proteins = []
    for idx, row in df.iterrows():
        proteins.append({
            "protein_id":    f"P{idx}",
            "sequence":      str(row["sequence"]),
            "free_energies": np.array(ast.literal_eval(row["free_energy"])),
        })
    return proteins


def load_timepoints() -> np.ndarray:
    tp = pd.read_csv(DATADIR / "timepoints.txt")["Timepoints"].values.astype(float)
    return tp  # shape (28,), seconds, starts with 0.0


def get_exchangeable_mask(sequence: str) -> np.ndarray:
    """Boolean mask for HDX-active amide NHs (skip N-term 2 residues and Pro)."""
    return np.array([i >= 2 and sequence[i] != 'P' for i in range(len(sequence))])


def compute_ph_rates(sequence: str, free_energies: np.ndarray, pH: float) -> np.ndarray:
    """Compute linear exchange rates (s^-1) for all residues at a given pH.

    Uses dG_open to get K_op, then scales by Milne intrinsic rate k_int(pH).
    Non-exchangeable residues get rate=0.

    Args:
        sequence: full amino acid sequence
        free_energies: dG_open (kcal/mol) for each exchangeable residue
                       length must equal get_exchangeable_mask(sequence).sum()
        pH: target pH

    Returns:
        (len(sequence),) array of linear rates s^-1, 0 for non-exchangeable
    """
    mask = get_exchangeable_mask(sequence)
    assert mask.sum() == len(free_energies), (
        f"Mask sum {mask.sum()} != free_energies length {len(free_energies)}"
    )
    k_int = milne_intrinsic_rates(sequence, TEMPERATURE, pH, D2O_PURITY, D2O_FRACTION)
    dg = np.clip(free_energies, 0.0, None)
    K_op = np.exp(-dg / (R_KCAL * TEMPERATURE))
    k_hdx = np.zeros(len(sequence))
    k_hdx[mask] = K_op * k_int[mask]
    return k_hdx  # (N,) linear s^-1


def simulate_distributions(
    undeuterated_dist: np.ndarray,
    timepoints: np.ndarray,
    k_hdx: np.ndarray,
) -> np.ndarray:
    """Compute (T, B) theoretical isotopic distributions. No backexchange.

    Args:
        undeuterated_dist: (B,) undeuterated isotope envelope (max-normalized)
        timepoints: (T,) time points in seconds
        k_hdx: (N,) per-residue linear exchange rates s^-1 (0 for non-exchangeable)

    Returns:
        (T, B) array of max-normalized isotopic distributions
    """
    inv_bx = np.ones(len(timepoints), dtype=np.float32)
    dists = theoretical_dist_all_timepoints(
        undeuterated_dist=jnp.asarray(undeuterated_dist, dtype=jnp.float32),
        timepoints=jnp.asarray(timepoints, dtype=jnp.float32),
        rates=jnp.asarray(k_hdx, dtype=jnp.float32),
        inv_backexchange_array=jnp.asarray(inv_bx),
        d2o_fraction=D2O_FRACTION,
        d2o_purity=D2O_PURITY,
        num_bins=NUM_BINS,
    )
    return np.asarray(dists)  # (T, B)


def compute_centroids(dists: np.ndarray) -> np.ndarray:
    """Compute mean deuterium content per timepoint.

    Args:
        dists: (T, B) isotopic distributions (any normalization)

    Returns:
        (T,) centroid positions (weighted mean bin index)
    """
    bins = np.arange(dists.shape[1], dtype=float)
    norm = dists / (dists.sum(axis=1, keepdims=True) + 1e-12)
    return (norm * bins).sum(axis=1)  # (T,)


def plot_timecourse(
    protein_id: str,
    dists_by_ph: dict[float, np.ndarray],
    timepoints: np.ndarray,
) -> None:
    """Save 28-row × 3-col grid of isotopic distributions per pH/timepoint."""
    T = len(timepoints)
    fig, axes = plt.subplots(
        T, 3,
        figsize=(9, T * 0.7),
        sharex=True,
        sharey="row",
    )
    bins = np.arange(NUM_BINS)

    for col, pH in enumerate(PHS):
        dists = dists_by_ph[pH]
        axes[0, col].set_title(PH_LABELS[pH], fontsize=10, color=PH_COLORS[pH])
        for row in range(T):
            ax = axes[row, col]
            ax.fill_between(bins, dists[row], color=PH_COLORS[pH], alpha=0.6)
            ax.set_yticks([])
            if col == 0:
                label = "0s" if timepoints[row] == 0 else f"{int(timepoints[row])}s"
                ax.set_ylabel(label, fontsize=5, rotation=0, ha="right", va="center")

    fig.supxlabel("Mass bin", fontsize=9)
    fig.suptitle(f"{protein_id} — isotopic distributions by pH and time", fontsize=11)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / f"{protein_id}_timecourse.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / f'{protein_id}_timecourse.png'}")


def plot_centroids(
    protein_id: str,
    centroids_by_ph: dict[float, np.ndarray],
    timepoints: np.ndarray,
) -> None:
    """Save centroid timecourse plot with log10 time axis, 3 pH colors."""
    fig, ax = plt.subplots(figsize=(7, 4))
    tp_nonzero = timepoints[timepoints > 0]

    for pH in PHS:
        cents_nz = centroids_by_ph[pH][timepoints > 0]
        ax.plot(
            np.log10(tp_nonzero), cents_nz,
            color=PH_COLORS[pH], label=PH_LABELS[pH],
            marker="o", markersize=3, linewidth=1.5,
        )

    ax.set_xlabel("log₁₀(time / s)", fontsize=11)
    ax.set_ylabel("Mean deuterium content (bins)", fontsize=11)
    ax.set_title(f"{protein_id} — HDX centroid timecourse by pH", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{protein_id}_centroids.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / f'{protein_id}_centroids.png'}")


def run_mcmc_per_ph(
    protein_id: str,
    sequence: str,
    timepoints: np.ndarray,
    dists_by_ph: dict[float, np.ndarray],
    undeuterated_dist: np.ndarray,
    config: MCMCConfig,
) -> dict[float, dict]:
    """Run independent single-pH MCMC for each pH condition.

    Returns dict: pH -> {"posterior": PosteriorRates, "pred_dists": ndarray, "rmse_tp": ndarray}
    """
    results = {}
    for pH in PHS:
        print(f"  [{protein_id}] Fitting pH {pH}...")
        pg = ProteinGroup(
            protein_id=protein_id,
            sequence=sequence,
            pHs=[pH],
            timepoints_by_pH={pH: timepoints},
            distributions_by_pH={pH: dists_by_ph[pH]},
            d2o_fraction=D2O_FRACTION,
            d2o_purity=D2O_PURITY,
            temperature=TEMPERATURE,
        )
        bx = BackExchangeResult(
            backexchange_value=0.0,
            backexchange_array=np.zeros(len(timepoints)),
            fit_rmse=0.0,
            theoretical_dist_at_saturation=undeuterated_dist,
        )
        posterior, _, diagnostics, pred_dists, rmse_tp = fit_rates(
            protein_group=pg,
            backexchange_by_pH={pH: bx},
            config=config,
            undeuterated_dist=undeuterated_dist,
            num_bins=NUM_BINS,
        )
        results[pH] = {
            "posterior": posterior,
            "pred_dists": pred_dists,
            "rmse_tp": rmse_tp,
        }
        print(f"    Done. Mean RMSE: {rmse_tp.mean():.4f}")
    return results


def true_log_rates_at_ph(sequence: str, free_energies: np.ndarray, pH: float) -> np.ndarray:
    """Return sorted natural-log exchange rates for exchangeable residues at a given pH.

    Returns:
        (N_exchangeable,) sorted ascending natural log rates
    """
    k = compute_ph_rates(sequence, free_energies, pH)
    k_ex = k[k > 0]
    return np.sort(np.log(k_ex))


def plot_rates_comparison(
    protein_id: str,
    sequence: str,
    free_energies: np.ndarray,
    mcmc_results: dict[float, dict],
) -> None:
    """Save 3-subplot figure: MCMC posterior vs true log-rates, one subplot per pH.

    Shows where each pH condition fails to recover rates:
    - pH 5/6: slow rates (left side) are poorly recovered (insufficient exchange)
    - pH 6/9: fast rates (right side) are poorly recovered (already fully exchanged)
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"{protein_id} — MCMC posterior vs true log-rates", fontsize=12)

    for ax, pH in zip(axes, PHS):
        true_lr = true_log_rates_at_ph(sequence, free_energies, pH)
        post_lr = mcmc_results[pH]["posterior"].mean
        ci5     = mcmc_results[pH]["posterior"].ci_5
        ci95    = mcmc_results[pH]["posterior"].ci_95

        n = min(len(true_lr), len(post_lr))
        true_lr = true_lr[:n]
        post_lr = post_lr[:n]
        ci5     = ci5[:n]
        ci95    = ci95[:n]

        yerr = np.array([np.maximum(post_lr - ci5, 0), np.maximum(ci95 - post_lr, 0)])
        ax.errorbar(
            true_lr, post_lr, yerr=yerr,
            fmt="o", color=PH_COLORS[pH], alpha=0.7,
            markersize=4, elinewidth=0.8, capsize=2,
            label="posterior mean ± 90% CI",
        )
        all_vals = np.concatenate([true_lr, post_lr])
        lims = [all_vals.min() - 1, all_vals.max() + 1]
        ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True ln(k) [s⁻¹]", fontsize=10)
        ax.set_ylabel("Inferred ln(k) [s⁻¹]", fontsize=10)
        ax.set_title(PH_LABELS[pH], fontsize=11, color=PH_COLORS[pH])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(FIGDIR / f"{protein_id}_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / f'{protein_id}_rates.png'}")


def run_mcmc_combined(
    protein_id: str,
    sequence: str,
    timepoints: np.ndarray,
    dists_by_ph: dict[float, np.ndarray],
    undeuterated_dist: np.ndarray,
    config: MCMCConfig,
) -> dict:
    """Run joint MCMC on pH 6 + pH 9 data simultaneously using the merge model.

    Rates are inferred at pH 6 (reference). pH 9 data is accounted for via a
    merge_fac parameter (log10-scale rate ratio), prior mean = 3.0.

    Returns dict with keys: posterior, merge_fac, pred_dists_ph6, pred_dists_ph9,
    rmse_ph6, rmse_ph9.
    """
    combined_phs = [6.0, 9.0]
    print(f"  [{protein_id}] Fitting combined pH 6 + pH 9...")
    pg = ProteinGroup(
        protein_id=protein_id,
        sequence=sequence,
        pHs=combined_phs,
        timepoints_by_pH={pH: timepoints for pH in combined_phs},
        distributions_by_pH={pH: dists_by_ph[pH] for pH in combined_phs},
        d2o_fraction=D2O_FRACTION,
        d2o_purity=D2O_PURITY,
        temperature=TEMPERATURE,
    )
    bx = BackExchangeResult(
        backexchange_value=0.0,
        backexchange_array=np.zeros(len(timepoints)),
        fit_rmse=0.0,
        theoretical_dist_at_saturation=undeuterated_dist,
    )
    posterior, merge_fac, diagnostics, pred_dists_all, rmse_all = fit_rates(
        protein_group=pg,
        backexchange_by_pH={pH: bx for pH in combined_phs},
        config=config,
        undeuterated_dist=undeuterated_dist,
        num_bins=NUM_BINS,
        reference_pH=6.0,
    )
    T = len(timepoints)
    # pred_dists_all is (2T, B): first T rows = pH 6 (reference), next T = pH 9
    pred_dists_ph6 = pred_dists_all[:T]
    pred_dists_ph9 = pred_dists_all[T:]
    rmse_ph6 = rmse_all[:T]
    rmse_ph9 = rmse_all[T:]
    print(f"    Done. RMSE pH6={rmse_ph6.mean():.4f}  pH9={rmse_ph9.mean():.4f}  "
          f"merge_fac={merge_fac.mean[0]:.3f}")
    return {
        "posterior": posterior,
        "merge_fac": merge_fac,
        "pred_dists_ph6": pred_dists_ph6,
        "pred_dists_ph9": pred_dists_ph9,
        "rmse_ph6": rmse_ph6,
        "rmse_ph9": rmse_ph9,
    }


def plot_combined_rates(
    protein_id: str,
    sequence: str,
    free_energies: np.ndarray,
    combined_result: dict,
    single_ph_results: dict[float, dict],
) -> None:
    """Save 2-subplot figure comparing combined pH6+9 posterior vs single-pH posteriors.

    Left: combined pH 6+9 posterior vs true rates at pH 6.
    Right: single-pH pH 6 and pH 9 posteriors overlaid vs true rates at pH 6,
           to show how combining the two pHs improves coverage.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{protein_id} — Combined pH 6+9 MCMC  "
        f"(merge_fac={combined_result['merge_fac'].mean[0]:.3f})",
        fontsize=12,
    )

    true_lr = true_log_rates_at_ph(sequence, free_energies, 6.0)

    # ── Left: combined posterior vs true at pH 6 ─────────────────────────────
    ax = axes[0]
    post_lr = combined_result["posterior"].mean
    ci5     = combined_result["posterior"].ci_5
    ci95    = combined_result["posterior"].ci_95
    n = min(len(true_lr), len(post_lr))
    tl, pl, c5, c9 = true_lr[:n], post_lr[:n], ci5[:n], ci95[:n]
    yerr = np.array([np.maximum(pl - c5, 0), np.maximum(c9 - pl, 0)])
    ax.errorbar(tl, pl, yerr=yerr, fmt="o", color="mediumpurple", alpha=0.75,
                markersize=4, elinewidth=0.8, capsize=2, label="pH 6+9 combined ± 90% CI")
    lims = [min(tl.min(), pl.min()) - 1, max(tl.max(), pl.max()) + 1]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True ln(k) at pH 6 [s⁻¹]", fontsize=10)
    ax.set_ylabel("Inferred ln(k) at pH 6 [s⁻¹]", fontsize=10)
    ax.set_title("Combined pH 6+9", fontsize=11, color="mediumpurple")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    # ── Right: single-pH pH 6 and pH 9 vs true at pH 6 ──────────────────────
    ax = axes[1]
    for pH, color, label in [
        (6.0, PH_COLORS[6.0], "pH 6 only"),
        (9.0, PH_COLORS[9.0], "pH 9 only (rates shifted to pH 6)"),
    ]:
        true_at_ph = true_log_rates_at_ph(sequence, free_energies, pH)
        post = single_ph_results[pH]["posterior"]
        n2 = min(len(true_lr), len(post.mean))
        pl2 = post.mean[:n2]
        # For pH 9 rates: subtract the expected pH offset to put them on pH 6 scale
        if pH == 9.0:
            offset = np.log(10) * combined_result["merge_fac"].mean[0]
            pl2 = pl2 - offset
        c5_2  = np.maximum(post.mean[:n2] - post.ci_5[:n2], 0)
        c9_2  = np.maximum(post.ci_95[:n2] - post.mean[:n2], 0)
        if pH == 9.0:
            c5_2  = np.maximum(pl2 - (post.ci_5[:n2] - offset), 0)
            c9_2  = np.maximum((post.ci_95[:n2] - offset) - pl2, 0)
        ax.errorbar(true_lr[:n2], pl2, yerr=np.array([c5_2, c9_2]),
                    fmt="o", color=color, alpha=0.6, markersize=4,
                    elinewidth=0.8, capsize=2, label=label)
    lims2 = [true_lr.min() - 1, true_lr.max() + 1]
    ax.plot(lims2, lims2, "k--", linewidth=1)
    ax.set_xlim(lims2); ax.set_ylim(lims2)
    ax.set_xlabel("True ln(k) at pH 6 [s⁻¹]", fontsize=10)
    ax.set_ylabel("Inferred ln(k) at pH 6 [s⁻¹]", fontsize=10)
    ax.set_title("Single-pH fits (projected to pH 6)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(FIGDIR / f"{protein_id}_combined_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / f'{protein_id}_combined_rates.png'}")


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    proteins   = load_data()
    timepoints = load_timepoints()
    config     = MCMCConfig(num_chains=4, num_warmup=100, num_samples=250)

    for p in proteins:
        pid = p["protein_id"]
        print(f"\n{'='*50}\nProcessing {pid}\n{'='*50}")

        undeu = theoretical_isotope_dist(p["sequence"], NUM_BINS)

        dists_by_ph = {
            pH: simulate_distributions(
                undeu, timepoints,
                compute_ph_rates(p["sequence"], p["free_energies"], pH),
            )
            for pH in PHS
        }

        plot_timecourse(pid, dists_by_ph, timepoints)

        cents_by_ph = {pH: compute_centroids(dists_by_ph[pH]) for pH in PHS}
        plot_centroids(pid, cents_by_ph, timepoints)

        mcmc_results = run_mcmc_per_ph(
            pid, p["sequence"], timepoints, dists_by_ph, undeu, config
        )

        plot_rates_comparison(pid, p["sequence"], p["free_energies"], mcmc_results)

        combined = run_mcmc_combined(
            pid, p["sequence"], timepoints, dists_by_ph, undeu, config
        )

        plot_combined_rates(pid, p["sequence"], p["free_energies"], combined, mcmc_results)

    print(f"\nDone. Figures saved to: {FIGDIR}")


if __name__ == "__main__":
    main()
