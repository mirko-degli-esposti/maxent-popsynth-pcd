"""
run_A0_toy.py
-------------
Experiment A0: Gibbs conditionals sanity check (K=6).

Validates that GibbsPCDSolver converges to the exact MaxEnt solution
on a small problem where exact enumeration is feasible (|X| = 216).

Results:
    - MRE decreases monotonically with N_pool (O(1/sqrt(N)) floor)
    - KL divergence scales as O(1/N)
    - ||Δλ||/||λ*|| remains constant (~0.34) due to gauge non-identifiability
      from unary constraints — MRE and KL are the correct diagnostics

Output:
    figures/A0_toy.png

Usage:
    python experiments/run_A0_toy.py
    python experiments/run_A0_toy.py --outdir results/figures
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import (ConstraintSet, GibbsPCDSolver, ExactMaxEntSolver,
                 WuGenerator, Evaluator,
                 plot_lambda_scatter, print_summary_table)


def run_A0(outdir: str = 'figures', seed: int = 0):

    os.makedirs(outdir, exist_ok=True)

    print("=" * 65)
    print("EXPERIMENT A0: Toy sanity check (K=6)")
    print("=" * 65)

    # ── Data generation ───────────────────────────────────────────
    wu0 = WuGenerator(K=6, domain_range=(2, 3), n_patterns=3,
                      pattern_arity=2, seed=seed)
    print(wu0.describe())
    print(f"\n  |X| = {np.prod(wu0.domain_sizes)}")

    data0 = wu0.generate(n_samples=100_000)
    cs0   = wu0.extract_constraints(data0)
    print(f"\n{cs0.summary()}")

    # ── Exact L-BFGS reference ────────────────────────────────────
    print("\n--- Exact L-BFGS ---")
    exact0 = ExactMaxEntSolver(cs0, verbose=True)
    exact0.fit(verbose=True)

    # ── GibbsPCDSolver at four pool sizes ─────────────────────────
    print("\n--- GibbsPCDSolver (varying N_pool) ---")
    N_POOLS    = [500, 2_000, 10_000, 100_000]
    N_SWEEPS   = 3
    LR         = 0.05
    N_OUTER    = 300

    gibbs0_results = {}
    for N_pool in N_POOLS:
        g = GibbsPCDSolver(cs0)
        g.fit(N_pool=N_pool, n_outer=N_OUTER, n_gibbs_sweeps=N_SWEEPS,
              lr=LR, seed=1, verbose_every=0)
        gibbs0_results[(N_pool, N_SWEEPS)] = g
        print(f"  N_pool={N_pool:>7,}:  MRE={g.final_mre:.5f}  "
              f"t={g.fit_time:.1f}s")

    print_summary_table("A0 — K=6 Toy", exact0, gibbs0_results)

    # ── Figures ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#2980b9']

    # (a) MRE convergence curves
    for (key, solver), color in zip(gibbs0_results.items(), colors):
        N_pool, _ = key
        mres = [h['mre'] for h in solver.history]
        axes[0].plot(mres, color=color, lw=1.5, label=f'N={N_pool:,}')
    axes[0].axhline(exact0.final_mre, color='k', ls='--', lw=1.5,
                    label='Exact L-BFGS')
    axes[0].set_xlabel('Outer iteration')
    axes[0].set_ylabel('MRE')
    axes[0].set_title('A0 — MRE convergence')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=9)

    # (b) lambda scatter (N=10K)
    best0 = gibbs0_results[(10_000, N_SWEEPS)]
    plot_lambda_scatter(exact0, best0,
                        title='A0 — lambda recovery (N=10K)',
                        ax=axes[1])

    # (c) alpha_hat vs alpha target
    alpha_hat0 = np.array([h['alpha_hat'] for h in best0.history])[-1]
    axes[2].scatter(cs0.alphas_array, alpha_hat0,
                    alpha=0.5, s=20, color='#9b59b6')
    m0 = max(cs0.alphas_array.max(), alpha_hat0.max()) * 1.05
    axes[2].plot([0, m0], [0, m0], 'k--', lw=1)
    axes[2].set_xlabel('alpha_j target')
    axes[2].set_ylabel('alpha_hat_j (pool)')
    axes[2].set_title('A0 — Frequencies: target vs estimate')

    plt.tight_layout()
    out_path = os.path.join(outdir, 'A0_toy.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")

    return gibbs0_results, exact0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment A0 — K=6 toy')
    parser.add_argument('--outdir', default='figures',
                        help='Output directory for figures (default: figures)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for WuGenerator (default: 0)')
    args = parser.parse_args()
    run_A0(outdir=args.outdir, seed=args.seed)
