"""
run_AISTAT_sensitivity.py
-------------------------
Experiment A-ISTAT-3: Pool-size sensitivity on Syn-ISTAT (K=15).

Sweeps N_pool in {10K, 25K, 50K, 100K, 200K} on the full 31-constraint
Syn-ISTAT problem. Raking at 100K serves as baseline.

Key findings (paper Table 5):
    - N=25K sweet spot: MRE=0.050 in 7 min
    - N=50K and N=100K give nearly identical MRE (stopping rule artefact)
    - Observed MRE lies above 1/(2*sqrt(N)) floor at all N,
      confirming mixing bias dominates over gradient variance at K=15

Output:
    Console table + figures/A_ISTAT_3_pareto.png

Usage:
    python experiments/run_AISTAT_sensitivity.py
    python experiments/run_AISTAT_sensitivity.py --outdir results/figures
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.syn_istat import build_syn_istat_constraint_sets
from experiments.helpers_synistat import (
    run_gibbs_std, run_raking, print_comparison
)


def run_aistat_sensitivity(outdir: str = 'figures'):

    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("  A-ISTAT-3: Pool-size sensitivity (K=15, cs_full)")
    print("=" * 70)

    cs_full, _, _ = build_syn_istat_constraint_sets()

    N_grid = [10_000, 25_000, 50_000, 100_000, 200_000]

    # ── GibbsPCDSolver sweep ──────────────────────────────────────
    rows_3 = []
    for N in N_grid:
        r = run_gibbs_std(cs_full, N_pool=N, seed=42,
                          label=f'Gibbs N={N//1000}K',
                          verbose_every=0)
        rows_3.append(r)
        print(f"  N={N:>7,}  MRE={r['MRE']:.5f}  "
              f"iters={r['n_iters']}  t={r['fit_time']:.1f}s")

    # ── Raking at 100K as baseline ────────────────────────────────
    print("\n--- Raking (N=100K, reference) ---")
    res_r1 = run_raking(cs_full, N_pop=100_000, tol=0.02, window=30,
                         seed=42, label='Raking N=100K')
    rows_3.append(res_r1)

    print_comparison(rows_3, title='A-ISTAT-3: Pool-size sensitivity (K=15)')

    # ── Figure ────────────────────────────────────────────────────
    gibbs_rows = [r for r in rows_3 if r['method'] == 'gibbs']
    raking_row = next(r for r in rows_3 if r['method'] == 'raking')

    ns    = [r['N_pool']   for r in gibbs_rows]
    mres  = [r['MRE']      for r in gibbs_rows]
    times = [r['fit_time'] for r in gibbs_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) MRE vs N_pool
    ax = axes[0]
    ax.plot(ns, mres, 'o-', color='steelblue', lw=2, ms=7,
            label='GibbsPCDSolver')
    ax.axhline(raking_row['MRE'], ls='--', color='tomato', lw=1.8,
               label=f"Raking (N={raking_row['N_pool']//1000}K)")
    ns_fine = np.logspace(np.log10(min(ns)), np.log10(max(ns)), 100)
    ax.plot(ns_fine, 0.5 / np.sqrt(ns_fine), ':',
            color='gray', lw=1.2, label='floor 1/(2√N)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N_pool')
    ax.set_ylabel('MRE')
    ax.set_title('A-ISTAT-3: MRE vs N_pool  (K=15)')
    ax.legend(fontsize=9)

    # (b) Pareto frontier
    ax = axes[1]
    sc = ax.scatter(times, mres, c=np.log10(ns),
                    cmap='Blues', s=90, zorder=3, vmin=3.5, vmax=5.5)
    for r in gibbs_rows:
        ax.annotate(f"N={r['N_pool']//1000}K",
                    (r['fit_time'], r['MRE']),
                    textcoords='offset points', xytext=(6, 3), fontsize=8)
    ax.axvline(raking_row['fit_time'], ls='--', color='tomato',
               lw=1.2, alpha=0.7)
    ax.scatter([raking_row['fit_time']], [raking_row['MRE']],
               color='tomato', marker='D', s=90, zorder=3,
               label=f"Raking N=100K (MRE=0 by construction†)")
    ax.set_xlabel('Fit time (s)')
    ax.set_ylabel('MRE')
    ax.set_yscale('log')
    ax.set_title('A-ISTAT-3: Pareto frontier  (K=15)')
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=axes[1], label='log10(N_pool)')

    plt.tight_layout()
    out_path = os.path.join(outdir, 'A_ISTAT_3_pareto.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")
    print("† Raking MRE=0 on training is exact by algebraic construction "
          "and not directly comparable to stochastic Gibbs estimates.")

    return rows_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A-ISTAT-3: Pool-size sensitivity')
    parser.add_argument('--outdir', default='figures')
    args = parser.parse_args()
    run_aistat_sensitivity(outdir=args.outdir)
