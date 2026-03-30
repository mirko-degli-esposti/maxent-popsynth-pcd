"""
run_AISTAT_diversity.py
-----------------------
Experiment A-ISTAT-DIV: Population diversity on Syn-ISTAT (K=15).

Compares GibbsPCDSolver and raking on demographic diversity metrics:
    1. N_eff = 1/sum(w^2) — effective sample size
    2. Unique profiles (distinct 15-attribute tuples)
    3. Empirical Shannon entropy H = -sum p(x) log p(x)
    4. Gini coefficient of raking weights

Key findings (paper Table 4):
    - GibbsPCDSolver: N_eff = N = 100,000 (uniform weights)
    - Raking: N_eff = 1,152 (1.2% of N), Gini = 0.951
    - Entropy advantage: 3.15 nats (86.8x in effective diversity)

Output:
    Console table + figures/A_ISTAT_3_diversity.png

Usage:
    python experiments/run_AISTAT_diversity.py
    python experiments/run_AISTAT_diversity.py --outdir results/figures
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.syn_istat import build_syn_istat_constraint_sets, DOMAIN_SIZES_SYNTH
from experiments.helpers_synistat import (
    run_gibbs_std, run_raking, empirical_entropy, gini
)


def run_aistat_diversity(outdir: str = 'figures'):

    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("  A-ISTAT-DIV: Population diversity (K=15, N=100K)")
    print("=" * 70)

    cs_full, _, _ = build_syn_istat_constraint_sets()
    N = 100_000

    # ── Full training (31 constraints) ───────────────────────────
    print("\n--- GibbsPCDSolver (full training, 31 constraints) ---")
    res_g1 = run_gibbs_std(cs_full, N_pool=N, seed=42,
                            label='K=15 Gibbs full', verbose_every=100)

    print("\n--- Raking (full training, 31 constraints) ---")
    res_r1 = run_raking(cs_full, N_pop=N, tol=0.02, window=30,
                         seed=42, label='K=15 Raking full')

    # ── Diversity metrics ─────────────────────────────────────────
    pool_g = res_g1['solver'].pool_          # (N, 15)
    pop_r  = res_r1['solver']._population   # (N, 15)
    wts_r  = res_r1['solver'].weights
    wts_r  = wts_r / wts_r.sum()

    # 1. N_eff
    N_eff_gibbs  = float(N)
    N_eff_raking = float(1.0 / np.sum(wts_r ** 2))

    # 2. Unique profiles
    rows_g   = [tuple(pool_g[i]) for i in range(N)]
    unique_g = len(set(rows_g))

    thresh   = 1.0 / (10 * N)
    active_idx = np.where(wts_r > thresh)[0]
    rows_r_active = [tuple(pop_r[i]) for i in active_idx]
    unique_r = len(set(rows_r_active))

    # 3. Shannon entropy
    H_gibbs  = empirical_entropy(rows_g)
    rows_r_all = [tuple(pop_r[i]) for i in range(N)]
    H_raking = empirical_entropy(rows_r_all, wts_r)
    H_max    = float(np.log(np.prod(DOMAIN_SIZES_SYNTH)))

    # 4. Gini
    gini_raking = gini(wts_r)
    gini_gibbs  = gini(np.ones(N) / N)

    # ── Print results ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Effective sample size (N={N:,})")
    print(f"{'='*60}")
    print(f"  Gibbs  N_eff = {N_eff_gibbs:>10,.0f}  ({N_eff_gibbs/N*100:.1f}% of N)")
    print(f"  Raking N_eff = {N_eff_raking:>10,.0f}  ({N_eff_raking/N*100:.1f}% of N)")
    print(f"  Ratio Gibbs/Raking = {N_eff_gibbs/N_eff_raking:.1f}x")

    print(f"\n{'='*60}")
    print(f"  Unique demographic profiles (K=15)")
    print(f"{'='*60}")
    print(f"  Gibbs  : {unique_g:>8,} unique profiles out of {N:,}")
    print(f"  Raking : {unique_r:>8,} active profiles (w > {thresh:.2e})")
    print(f"  Ratio Gibbs/Raking = {unique_g/max(unique_r,1):.1f}x")

    print(f"\n{'='*60}")
    print(f"  Empirical Shannon entropy  (H_max = {H_max:.2f} nats)")
    print(f"{'='*60}")
    print(f"  Gibbs  H = {H_gibbs:.4f}  ({H_gibbs/H_max*100:.1f}% of H_max)")
    print(f"  Raking H = {H_raking:.4f}  ({H_raking/H_max*100:.1f}% of H_max)")
    print(f"  Difference = {H_gibbs - H_raking:.4f} nats")

    print(f"\n{'='*60}")
    print(f"  Gini coefficient  (0=uniform, 1=concentrated)")
    print(f"{'='*60}")
    print(f"  Gibbs  Gini = {gini_gibbs:.6f}  (uniform by construction)")
    print(f"  Raking Gini = {gini_raking:.4f}")

    # ── Figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    # (a) Raking weight distribution
    ax = axes[0]
    sorted_w = np.sort(wts_r)[::-1]
    ax.semilogy(np.arange(1, N+1), sorted_w, color='#e74c3c', lw=1.2)
    ax.axhline(1.0/N, color='#2c7bb6', ls='--', lw=1.5,
               label=f'Gibbs uniform (1/N)')
    ax.set_xlabel('Individual rank')
    ax.set_ylabel('Weight (log scale)')
    ax.set_title('Weight distribution')
    ax.legend(fontsize=9)

    # (b) Lorenz curve of raking weights
    ax = axes[1]
    sw = np.sort(wts_r)
    cumw = np.cumsum(sw)
    cumw /= cumw[-1]
    frac = np.linspace(0, 1, N)
    ax.plot(frac, cumw, color='#e74c3c', lw=1.5,
            label=f'Raking (Gini={gini_raking:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect equality')
    ax.fill_between(frac, frac, cumw, alpha=0.15, color='#e74c3c')
    ax.set_xlabel('Cumulative share of individuals')
    ax.set_ylabel('Cumulative share of weight')
    ax.set_title('Lorenz curve (raking weights)')
    ax.legend(fontsize=9)

    # (c) Normalised diversity metrics comparison
    ax = axes[2]
    metrics = ['N_eff / N', 'Unique / N', 'H / H_max']
    gibbs_vals  = [N_eff_gibbs/N, unique_g/N, H_gibbs/H_max]
    raking_vals = [N_eff_raking/N, unique_r/N, H_raking/H_max]
    x = np.arange(len(metrics))
    w = 0.35
    bars_g = ax.bar(x - w/2, gibbs_vals,  w, label='Gibbs PCD',
                    color='#2c7bb6', alpha=0.85)
    bars_r = ax.bar(x + w/2, raking_vals, w, label='Raking',
                    color='#e74c3c', alpha=0.85)
    for i, (gv, rv) in enumerate(zip(gibbs_vals, raking_vals)):
        ratio = gv / max(rv, 1e-9)
        ax.text(i, max(gv, rv) + 0.02, f'{ratio:.1f}x',
                ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel('Normalised value')
    ax.set_title('Diversity metrics (normalised)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.25)

    plt.tight_layout()
    out_path = os.path.join(outdir, 'A_ISTAT_3_diversity.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")

    return res_g1, res_r1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A-ISTAT-DIV: Population diversity')
    parser.add_argument('--outdir', default='figures')
    parser.add_argument('--N_pool', type=int, default=100_000)
    args = parser.parse_args()
    run_aistat_diversity(outdir=args.outdir)
