"""
run_A1b_planted_k10.py
----------------------
Experiment A1b: Planted exponential family (K=10).

Uses PlantedExpFamilyGenerator to construct a ground truth with
analytically known lambda* and exact p_{lambda*}. This eliminates
gauge non-identifiability (no unary constraints) and enables exact
computation of KL(p_{lambda*} || p_{lambda_MCMC}) and
||lambda_MCMC - lambda*|| / ||lambda*||.

Key findings:
    - ||Δλ||/||λ*|| decreases from 0.072 (N=1K) to 0.016 (N=50K) — O(1/sqrt(N))
    - KL divergence scales as O(1/N)
    - lambda scatter at N=50K lies tightly around the diagonal (no gauge offset)

Output:
    figures/A1b_planted_k10.png

Usage:
    python experiments/run_A1b_planted_k10.py
    python experiments/run_A1b_planted_k10.py --outdir results/figures
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import (GibbsPCDSolver, ExactMaxEntSolver,
                 PlantedExpFamilyGenerator, Evaluator)


def run_A1b(outdir: str = 'figures', seed: int = 20):

    os.makedirs(outdir, exist_ok=True)

    print("=" * 65)
    print("EXPERIMENT A1b: Planted exponential family (K=10)")
    print("=" * 65)

    # ── Ground truth: K=10, all domains = 3, |X| = 59,049 ────────
    domain_sizes_b = [3] * 10
    gen_b = PlantedExpFamilyGenerator(domain_sizes_b, seed=seed)
    cs_b, lambdas_true_b = gen_b.plant_constraints(
        n_constraints=30, arity=2, lambda_scale=1.2
    )
    print(gen_b.describe())
    print(f"\n{cs_b.summary()}")
    print(f"  lambda* in [{lambdas_true_b.min():.3f}, "
          f"{lambdas_true_b.max():.3f}]")

    p_true_b = gen_b.probs   # exact p_{lambda*}

    # ── Exact L-BFGS reference ────────────────────────────────────
    print("\n--- Exact L-BFGS ---")
    exact_b = ExactMaxEntSolver(cs_b, verbose=True)
    exact_b.fit(verbose=False)
    kl_exact = Evaluator.kl(p_true_b, exact_b.get_probs())
    pd_exact = Evaluator.parameter_distance(exact_b.lambdas, lambdas_true_b)
    print(f"  MRE={exact_b.final_mre:.5f}  "
          f"||lambda_exact - lambda*||/||lambda*||={pd_exact:.5f}  "
          f"KL(p*||p_exact)={kl_exact:.6f}")

    # ── GibbsPCDSolver vs analytical lambda* ──────────────────────
    print("\n--- GibbsPCDSolver (vs analytical lambda*) ---")
    N_POOLS  = [1_000, 5_000, 10_000, 50_000]
    N_SWEEPS = 5
    LR       = 0.04

    gibbs_b_results = {}
    for N_pool in N_POOLS:
        g = GibbsPCDSolver(cs_b)
        g.fit(N_pool=N_pool, n_outer=500, n_gibbs_sweeps=N_SWEEPS,
              lr=LR, seed=2, verbose_every=0)
        gibbs_b_results[(N_pool, N_SWEEPS)] = g

        p_g  = Evaluator.compute_gibbs_probs(g, exact_b)
        kl_g = Evaluator.kl(p_true_b, p_g)
        pd_g = Evaluator.parameter_distance(g.lambdas, lambdas_true_b)
        print(f"  N={N_pool:>7,}:  MRE={g.final_mre:.5f}  "
              f"||Δλ||/||λ*||={pd_g:.5f}  "
              f"KL={kl_g:.6f}  t={g.fit_time:.1f}s")

    # ── Figures ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ns_b  = [N for (N, s) in gibbs_b_results]
    kls_b = [Evaluator.kl(p_true_b,
              Evaluator.compute_gibbs_probs(gibbs_b_results[(N, N_SWEEPS)],
                                             exact_b))
             for N in ns_b]
    pds_b = [Evaluator.parameter_distance(
              gibbs_b_results[(N, N_SWEEPS)].lambdas, lambdas_true_b)
             for N in ns_b]

    # (a) KL and parameter distance vs N_pool (dual y-axis)
    ax  = axes[0]
    ax2 = ax.twinx()
    ax.loglog(ns_b, kls_b, 's-', color='#e74c3c', lw=1.5, ms=7, label='KL')
    ax2.loglog(ns_b, pds_b, 'o--', color='#3498db', lw=1.5, ms=7,
               label='||Δλ||/||λ*||')
    n_ref   = np.array(ns_b)
    kl_ref  = kls_b[0] * ns_b[0] / n_ref
    ax.loglog(n_ref, kl_ref, 'k:', lw=1, alpha=0.5)
    ax.set_xlabel('N_pool')
    ax.set_ylabel('KL(p* || p_MCMC)', color='#e74c3c')
    ax2.set_ylabel('||Δλ||/||λ*||', color='#3498db')
    ax.set_title('A1b — Error vs N_pool')
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc='upper right')

    # (b) lambda_MCMC vs lambda* scatter (N=50K)
    best_b = gibbs_b_results[(50_000, N_SWEEPS)]
    ax_sc  = axes[1]
    ax_sc.scatter(lambdas_true_b, best_b.lambdas,
                  alpha=0.5, s=25, color='#2ecc71')
    lim_b = np.abs(lambdas_true_b).max() * 1.15
    ax_sc.plot([-lim_b, lim_b], [-lim_b, lim_b], 'k--', lw=1)
    ax_sc.set_xlabel('lambda*')
    ax_sc.set_ylabel('lambda_MCMC')
    ax_sc.set_title('A1b — lambda vs lambda* (N=50K)')
    pd_best = Evaluator.parameter_distance(best_b.lambdas, lambdas_true_b)
    ax_sc.text(0.05, 0.92, f'||Δλ||/||λ*|| = {pd_best:.4f}',
               transform=ax_sc.transAxes, fontsize=9)

    # (c) Estimated constraint frequencies vs targets
    alpha_true_unary = np.array([cs_b.alphas[j] for j in range(cs_b.m)
                                  if cs_b.attrs_list[j].size == 1])
    alpha_hat_unary  = np.array([best_b.history[-1]['alpha_hat'][j]
                                  for j in range(cs_b.m)
                                  if cs_b.attrs_list[j].size == 1])
    alpha_true_bin   = np.array([cs_b.alphas[j] for j in range(cs_b.m)
                                  if cs_b.attrs_list[j].size == 2])
    alpha_hat_bin    = np.array([best_b.history[-1]['alpha_hat'][j]
                                  for j in range(cs_b.m)
                                  if cs_b.attrs_list[j].size == 2])

    if len(alpha_true_unary) > 0:
        axes[2].scatter(alpha_true_unary, alpha_hat_unary,
                        alpha=0.7, s=35, color='#9b59b6',
                        label='unary', zorder=3)
    axes[2].scatter(alpha_true_bin, alpha_hat_bin,
                    alpha=0.4, s=20, color='#3498db',
                    label='binary', zorder=2)
    all_vals = np.concatenate([alpha_true_bin, alpha_hat_bin])
    m_b = all_vals.max() * 1.05
    axes[2].plot([0, m_b], [0, m_b], 'k--', lw=1)
    axes[2].set_xlabel('alpha_j target')
    axes[2].set_ylabel('alpha_hat_j MCMC (N=50K)')
    axes[2].set_title('A1b — Frequencies: target vs estimate')
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(outdir, 'A1b_planted_k10.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")

    return gibbs_b_results, exact_b, gen_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment A1b — K=10 planted exp-family')
    parser.add_argument('--outdir', default='figures',
                        help='Output directory for figures (default: figures)')
    parser.add_argument('--seed', type=int, default=20,
                        help='Random seed (default: 20)')
    args = parser.parse_args()
    run_A1b(outdir=args.outdir, seed=args.seed)
