"""
run_A1a_wu_k8.py
----------------
Experiment A1a: Wu benchmark (K=8).

Validates GibbsPCDSolver against exact L-BFGS on a WuGenerator
instance with K=8 and binary planted patterns (|X| = 6,144).

Key findings:
    - MRE decreases from 0.099 (N=1K) to 0.011 (N=50K)
    - KL divergence scales as O(1/N)
    - ||Δλ||/||λ*|| ≈ 0.50 across all N (gauge non-identifiability)
    - Doubling sweeps s=5->10 at N=20K yields no MRE improvement

Output:
    figures/A1a_wu_k8.png

Usage:
    python experiments/run_A1a_wu_k8.py
    python experiments/run_A1a_wu_k8.py --outdir results/figures
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import (GibbsPCDSolver, ExactMaxEntSolver, WuGenerator,
                 Evaluator, plot_lambda_scatter, print_summary_table)


def run_A1a(outdir: str = 'figures', seed: int = 10):

    os.makedirs(outdir, exist_ok=True)

    print("=" * 65)
    print("EXPERIMENT A1a: Wu benchmark (K=8)")
    print("=" * 65)

    # ── Data generation ───────────────────────────────────────────
    wu1 = WuGenerator(K=8, domain_range=(2, 4), n_patterns=6,
                      pattern_arity=2, seed=seed)
    print(wu1.describe())

    data1 = wu1.generate(n_samples=200_000)
    cs1   = wu1.extract_constraints(data1)
    print(f"\n{cs1.summary()}")

    # ── Exact L-BFGS reference ────────────────────────────────────
    print("\n--- Exact L-BFGS ---")
    exact1 = ExactMaxEntSolver(cs1, verbose=True)
    exact1.fit(verbose=False)
    print(f"  MRE={exact1.final_mre:.5f}  t={exact1.fit_time:.2f}s")

    # ── GibbsPCDSolver — N_pool x n_sweeps grid ───────────────────
    print("\n--- GibbsPCDSolver ---")
    configs = [
        (1_000,  5),
        (5_000,  5),
        (20_000, 5),
        (50_000, 5),
        (20_000, 10),   # same N, more sweeps — control
    ]

    gibbs1_results = {}
    p_exact = exact1.get_probs()

    for (N_pool, n_sweeps) in configs:
        g = GibbsPCDSolver(cs1)
        g.fit(N_pool=N_pool, n_outer=400, n_gibbs_sweeps=n_sweeps,
              lr=0.05, tol=0.02, window=30, seed=1, verbose_every=0)
        gibbs1_results[(N_pool, n_sweeps)] = g

        p_g   = Evaluator.compute_gibbs_probs(g, exact1)
        pars  = Evaluator.parameter_distance(g.lambdas, exact1.lambdas)
        kl    = Evaluator.kl(p_exact, p_g)
        print(f"  N={N_pool:>6,} s={n_sweeps}:  MRE={g.final_mre:.5f}  "
              f"||Δλ||/||λ||={pars:.4f}  KL={kl:.5f}  t={g.fit_time:.1f}s")

    print_summary_table("A1a — K=8 Wu", exact1, gibbs1_results)

    # ── Figures ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) MRE convergence curves (s=5 runs)
    ns_plot = [(N, s) for (N, s) in gibbs1_results if s == 5]
    colors  = plt.cm.plasma(np.linspace(0.1, 0.85, len(ns_plot)))

    for (N_pool, n_sweeps), color in zip(ns_plot, colors):
        mres = [h['mre'] for h in gibbs1_results[(N_pool, n_sweeps)].history]
        axes[0].plot(mres, color=color, lw=1.5, label=f'N={N_pool:,}')
    axes[0].axhline(exact1.final_mre, color='k', ls='--', lw=1.5,
                    label='Exact L-BFGS')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Outer iteration')
    axes[0].set_ylabel('MRE')
    axes[0].set_title('A1a — MRE convergence')
    axes[0].legend(fontsize=8)

    # (b) lambda scatter (N=50K, s=5)
    best1 = gibbs1_results[(50_000, 5)]
    plot_lambda_scatter(exact1, best1,
                        title='A1a — lambda recovery (N=50K)',
                        ax=axes[1])

    # (c) Final MRE vs N_pool (s=5) with 1/sqrt(N) reference
    ns_final  = [N for (N, s) in ns_plot]
    mre_final = [gibbs1_results[(N, 5)].final_mre for N in ns_final]
    axes[2].loglog(ns_final, mre_final, 'o-', color='#e74c3c',
                   lw=1.5, ms=7, label='GibbsPCD')
    n_ref   = np.array(ns_final)
    mre_ref = mre_final[0] * np.sqrt(ns_final[0] / n_ref)
    axes[2].loglog(n_ref, mre_ref, 'k--', lw=1, alpha=0.5,
                   label='1/√N ref')
    axes[2].axhline(exact1.final_mre, color='k', ls=':', lw=1,
                    label='Exact L-BFGS')
    axes[2].set_xlabel('N_pool')
    axes[2].set_ylabel('Final MRE')
    axes[2].set_title('A1a — MRE vs N_pool')
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(outdir, 'A1a_wu_k8.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")

    return gibbs1_results, exact1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment A1a — K=8 Wu')
    parser.add_argument('--outdir', default='figures',
                        help='Output directory for figures (default: figures)')
    parser.add_argument('--seed', type=int, default=10,
                        help='Random seed for WuGenerator (default: 10)')
    args = parser.parse_args()
    run_A1a(outdir=args.outdir, seed=args.seed)
