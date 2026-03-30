"""
run_A1c_sensitivity.py
----------------------
Experiment A1c: Pool size x sweeps sensitivity (K=8).

Evaluates GibbsPCDSolver over a full N_pool x n_sweeps grid on the
Wu K=8 benchmark. Produces MRE and wall-clock heatmaps plus a
Pareto frontier of accuracy vs compute time.

Key findings:
    - For binary constraints: Pareto-optimal at N=50K, s=1
    - Extra sweeps do not help when gradient variance dominates
    - For ternary constraints: higher s yields better MRE floor

Output:
    figures/A1c_sensitivity.png
    figures/A1c_pareto.png

Usage:
    python experiments/run_A1c_sensitivity.py
    python experiments/run_A1c_sensitivity.py --outdir results/figures
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import GibbsPCDSolver, WuGenerator, ExactMaxEntSolver


def run_A1c(outdir='figures', seed=10):

    os.makedirs(outdir, exist_ok=True)

    print("=" * 65)
    print("EXPERIMENT A1c: Pool size x sweeps sensitivity (K=8)")
    print("=" * 65)

    wu1 = WuGenerator(K=8, domain_range=(2, 4), n_patterns=6,
                      pattern_arity=2, seed=seed)
    print(wu1.describe())
    data1 = wu1.generate(n_samples=200_000)
    cs1   = wu1.extract_constraints(data1)

    print("\n--- Exact L-BFGS (reference) ---")
    exact1 = ExactMaxEntSolver(cs1, verbose=False)
    exact1.fit(verbose=False)
    print(f"  MRE={exact1.final_mre:.5f}  t={exact1.fit_time:.2f}s")

    N_pool_grid  = [1_000, 5_000, 20_000, 50_000]
    n_sweep_grid = [1, 3, 5, 10]
    mre_grid     = np.full((len(N_pool_grid), len(n_sweep_grid)), np.nan)
    time_grid    = np.full((len(N_pool_grid), len(n_sweep_grid)), np.nan)

    total = len(N_pool_grid) * len(n_sweep_grid)
    run   = 0
    print(f"\n--- Grid search ({total} configurations) ---")

    for i, N_pool in enumerate(N_pool_grid):
        for j, n_sweeps in enumerate(n_sweep_grid):
            run += 1
            g = GibbsPCDSolver(cs1)
            g.fit(N_pool=N_pool, n_outer=400, n_gibbs_sweeps=n_sweeps,
                  lr=0.05, tol=0.0, seed=7, verbose_every=0)
            mre_grid[i, j]  = g.final_mre
            time_grid[i, j] = g.fit_time
            print(f"  [{run:2d}/{total}] N={N_pool:>6,} s={n_sweeps}:  "
                  f"MRE={g.final_mre:.5f}  t={g.fit_time:.1f}s")

    # Figure 1: heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im0 = axes[0].imshow(mre_grid, aspect='auto', cmap='RdYlGn_r', origin='lower')
    axes[0].set_xticks(range(len(n_sweep_grid))); axes[0].set_xticklabels(n_sweep_grid)
    axes[0].set_yticks(range(len(N_pool_grid))); axes[0].set_yticklabels([f'{n:,}' for n in N_pool_grid])
    axes[0].set_xlabel('n_sweeps'); axes[0].set_ylabel('N_pool')
    axes[0].set_title('A1c — Final MRE')
    plt.colorbar(im0, ax=axes[0])
    for i in range(len(N_pool_grid)):
        for j in range(len(n_sweep_grid)):
            axes[0].text(j, i, f'{mre_grid[i,j]:.4f}', ha='center', va='center',
                         fontsize=8, color='white' if mre_grid[i,j] > mre_grid.mean() else 'black')

    im1 = axes[1].imshow(time_grid, aspect='auto', cmap='Blues', origin='lower')
    axes[1].set_xticks(range(len(n_sweep_grid))); axes[1].set_xticklabels(n_sweep_grid)
    axes[1].set_yticks(range(len(N_pool_grid))); axes[1].set_yticklabels([f'{n:,}' for n in N_pool_grid])
    axes[1].set_xlabel('n_sweeps'); axes[1].set_ylabel('N_pool')
    axes[1].set_title('A1c — Wall-clock (s)')
    plt.colorbar(im1, ax=axes[1])
    for i in range(len(N_pool_grid)):
        for j in range(len(n_sweep_grid)):
            axes[1].text(j, i, f'{time_grid[i,j]:.1f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    out1 = os.path.join(outdir, 'A1c_sensitivity.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out1}")

    # Figure 2: Pareto frontier
    fig2, ax_p = plt.subplots(figsize=(7, 5))
    colors_p  = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_pool_grid)))
    markers_p = ['o', 's', '^', 'D']
    for i, (N_pool, color) in enumerate(zip(N_pool_grid, colors_p)):
        for j, (n_sw, marker) in enumerate(zip(n_sweep_grid, markers_p)):
            ax_p.scatter(time_grid[i, j], mre_grid[i, j],
                         color=color, marker=marker, s=80, zorder=3)
    ax_p.axhline(exact1.final_mre, color='k', ls='--', lw=1.5,
                 label=f'Exact MRE={exact1.final_mre:.4f}')
    ax_p.set_xlabel('Wall-clock (s)'); ax_p.set_ylabel('Final MRE')
    ax_p.set_title('A1c — Pareto frontier: accuracy vs time')
    ax_p.set_yscale('log')
    legend_n = [Patch(color=c, label=f'N={N:,}') for N, c in zip(N_pool_grid, colors_p)]
    legend_s = [Line2D([0],[0], marker=m, color='gray', ls='None',
                       markersize=7, label=f's={s}') for m, s in zip(markers_p, n_sweep_grid)]
    ax_p.legend(handles=legend_n + legend_s, fontsize=8, loc='upper right', ncol=2)
    plt.tight_layout()
    out2 = os.path.join(outdir, 'A1c_pareto.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out2}")

    return mre_grid, time_grid, N_pool_grid, n_sweep_grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment A1c — sensitivity grid')
    parser.add_argument('--outdir', default='figures')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    run_A1c(outdir=args.outdir, seed=args.seed)
