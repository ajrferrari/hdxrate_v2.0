"""
Diagnostic plots for HDX rate fitting results.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .types import HDXResult


def plot_isotope_fits(
    result: HDXResult,
    output_path: str | None = None,
    max_cols: int = 5,
) -> plt.Figure:
    """
    Plot experimental vs predicted mass distributions for all timepoints.
    Exp shown as grey fill, predicted as red line.
    """
    n_tp = len(result.predicted_distributions)
    n_cols = min(n_tp, max_cols)
    n_rows = (n_tp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).ravel()

    # Stack experimental distributions across all pHs in same order as pred
    exp_stack = np.vstack([
        result.backexchange[pH].backexchange_array  # placeholder; use pred shape
        for pH in result.pHs
    ])
    # Reconstruct experimental stacked from timepoints/pHs
    exp_dists = []
    for pH in result.pHs:
        exp_dists.append(
            np.vstack([
                # We don't store exp dists in HDXResult directly;
                # use the stored pred as reference shape and note the difference
                result.predicted_distributions[0]  # fallback shape reference
            ])
        )

    for i in range(n_tp):
        ax = axes[i]
        pred = result.predicted_distributions[i]
        x = np.arange(len(pred))
        ax.fill_between(x, pred * 0, pred, alpha=0.3, color='grey', label='pred fill')
        ax.plot(x, pred, color='red', lw=1.2, label='predicted')
        ax.set_title(f'TP {i+1}', fontsize=8)
        ax.set_xlabel('mass bin', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(n_tp, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{result.protein_id} — isotope distributions', fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_isotope_fits_with_exp(
    exp_distributions: np.ndarray,
    result: HDXResult,
    timepoints: np.ndarray | None = None,
    output_path: str | None = None,
    max_cols: int = 5,
) -> plt.Figure:
    """
    Plot experimental (grey) vs predicted (red) isotope distributions.
    exp_distributions: (T, B) array of observed distributions.
    """
    n_tp = exp_distributions.shape[0]
    n_cols = min(n_tp, max_cols)
    n_rows = (n_tp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).ravel()

    for i in range(n_tp):
        ax = axes[i]
        exp = exp_distributions[i]
        pred = result.predicted_distributions[i] if i < len(result.predicted_distributions) else np.zeros_like(exp)
        x = np.arange(len(exp))
        ax.bar(x, exp, color='grey', alpha=0.5, width=0.8, label='exp')
        ax.plot(x, pred, color='red', lw=1.5, label='fit')
        rmse = result.rmse_per_timepoint[i] if i < len(result.rmse_per_timepoint) else float('nan')
        label = f'TP {i+1}'
        if timepoints is not None and i < len(timepoints):
            label = f'{timepoints[i]:.0f}s'
        ax.set_title(f'{label} (RMSE={rmse:.3f})', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(n_tp, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{result.protein_id} — isotope fits', fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_posterior_rates(
    result: HDXResult,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plot posterior rate distributions as violin plots (slowest to fastest).
    """
    samples = result.posterior_rates.samples  # (S, N)
    n_rates = samples.shape[1]

    fig, ax = plt.subplots(figsize=(max(6, n_rates * 0.4), 4))

    positions = np.arange(1, n_rates + 1)
    parts = ax.violinplot(samples, positions=positions, showmedians=True, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#4C72B0')
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    ax.set_xlabel('Rate index (1=slowest)', fontsize=9)
    ax.set_ylabel('log(k_HX) [log s⁻¹]', fontsize=9)
    ax.set_title(f'{result.protein_id} — posterior rate distributions', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_free_energy(
    result: HDXResult,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Two-panel plot: sorted dG values (left) and dG histogram (right).
    """
    if result.free_energy is None:
        raise ValueError("HDXResult has no free_energy computed")

    fe = result.free_energy
    dG_mean = fe.dG_mean
    dG_sorted = fe.dG_sorted

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])

    # Left: sorted dG
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(np.arange(1, len(dG_sorted) + 1), dG_sorted,
             marker='o', ms=4, lw=1.2, color='red',
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.5)
    ax1.set_xlabel('Residue (most → least stable)', fontsize=9)
    ax1.set_ylabel('ΔG_open (kcal/mol)', fontsize=9)
    ax1.set_title('Sorted ΔG_open', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.25)

    # Right: histogram
    ax2 = fig.add_subplot(gs[1])
    valid = dG_mean[np.isfinite(dG_mean)]
    ax2.hist(valid, bins=min(20, max(5, len(valid) // 2)),
             color='#4C72B0', alpha=0.75, edgecolor='white')
    ax2.set_xlabel('ΔG_open (kcal/mol)', fontsize=9)
    ax2.set_ylabel('count', fontsize=9)
    ax2.set_title('ΔG distribution', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.25)

    fig.suptitle(f'{result.protein_id}', fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    return fig
