"""
run_A2_scaling.py
-----------------
Experiment A2: Scaling GibbsPCDSolver beyond exact enumeration.

Runs three sub-experiments:
    A2a  K=12  — transition regime (exact L-BFGS still available)
    A2b  K=20  — mixed regime (|X| not enumerable)
    A2c  K=30  — pure MCMC regime (beyond Pachet-Zucker boundary)

For K >= 20, exact enumeration is impossible and GibbsPCDSolver is
the only MaxEnt-based method that can operate. Raking is included
as baseline at all K values.

Key findings (paper Table 2):
    - GibbsPCDSolver: MRE in [0.010, 0.018] across all K
    - Raking N_eff/N collapses from 7.1% (K=12) to 0.8% (K=30)
    - Runtime scales as O(K): ~600s at K=12, ~965s at K=30

Output:
    Console tables (no figures — numerical results only)

Usage:
    python experiments/run_A2_scaling.py
    python experiments/run_A2_scaling.py --no_numba   # pure NumPy
    python experiments/run_A2_scaling.py --skip_a2c   # skip K=30
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import WuGenerator, ExactMaxEntSolver, Evaluator
from experiments.helpers_a2 import (
    run_gibbs_a2, run_raking,
    pool_diversity, raking_diversity,
    print_a2_table
)


# ------------------------------------------------------------------ #
#  A2a — K=12                                                          #
# ------------------------------------------------------------------ #

def run_a2a(use_numba=False):
    print("\n" + "="*70)
    print("  A2a — K=12  (transition, exact available)")
    print("="*70)

    wu12 = WuGenerator(K=12, domain_range=(2, 4), n_patterns=6,
                       pattern_arity=3, seed=20)
    print(wu12.describe())
    data12 = wu12.generate(n_samples=200_000)
    cs12   = wu12.extract_constraints(data12)
    X_size_12 = int(np.prod(wu12.domain_sizes))
    print(f"\n{cs12.summary()}")
    print(f"|X| = {X_size_12:,}")

    # Exact L-BFGS (if enumerable)
    exact12 = None
    if X_size_12 <= 5_000_000:
        print("\n--- Exact L-BFGS ---")
        exact12 = ExactMaxEntSolver(cs12, verbose=False)
        exact12.fit(verbose=False)
        print(f"  MRE={exact12.final_mre:.5f}  t={exact12.fit_time:.1f}s")
    else:
        print(f"\n  |X|={X_size_12:,} — exact solver skipped")

    # GibbsPCDSolver
    print("\n--- GibbsPCDSolver ---")
    a2a_rows = []
    configs = [
        (20_000,  1, 0.05),
        (50_000,  1, 0.05),
        (50_000,  3, 0.01),
        (100_000, 3, 0.01),
    ]
    for N, s, lr in configs:
        r = run_gibbs_a2(cs12, N_pool=N, n_gibbs_sweeps=s, lr=lr,
                         n_outer=600, tol=0.02, window=50,
                         seed=1, label=f'K=12 N={N//1000}K s={s}',
                         use_numba=use_numba)
        if exact12 is not None:
            p_exact = exact12.get_probs()
            p_gibbs = Evaluator.compute_gibbs_probs(r['solver'], exact12)
            r['KL']    = Evaluator.kl(p_exact, p_gibbs)
            r['pdist'] = Evaluator.parameter_distance(r['solver'].lambdas,
                                                       exact12.lambdas)
        else:
            r['KL'] = r['pdist'] = float('nan')

        pool12 = r['solver']._init_pool(N, seed=99)
        for _ in range(100):
            pool12 = r['solver']._gibbs_sweep(pool12, r['solver'].lambdas)
        r['neff'] = N
        r['n_unique'], r['H'] = pool_diversity(pool12)
        a2a_rows.append(r)
        print(f"  {r['label']}: MRE={r['MRE']:.5f}  "
              f"KL={r['KL']:.5f}  t={r['fit_time']:.1f}s")

    # Raking
    print("\n--- Raking ---")
    for N in [50_000, 100_000]:
        r = run_raking(cs12, N_pop=N, tol=0.02, window=30,
                       seed=1, label=f'K=12 Raking N={N//1000}K')
        neff_r, n_unique_r, H_r = raking_diversity(r['solver'])
        r['neff']     = neff_r
        r['n_unique'] = n_unique_r
        r['H']        = H_r
        a2a_rows.append(r)
        print(f"  {r['label']}: MRE={r['MRE']:.5f}  "
              f"N_eff={neff_r:.0f}  t={r['fit_time']:.1f}s")

    print_a2_table(a2a_rows, 'A2a — K=12 (ternary)')
    if exact12:
        print(f"  Exact reference: MRE={exact12.final_mre:.5f}")

    return a2a_rows


# ------------------------------------------------------------------ #
#  A2b — K=20                                                          #
# ------------------------------------------------------------------ #

def run_a2b(use_numba=True):
    print("\n" + "="*70)
    print("  A2b — K=20  (mixed regime, no exact)")
    print("="*70)

    wu20 = WuGenerator(K=20, domain_range=(2, 3), n_patterns=10,
                       pattern_arity=3, seed=30)
    print(wu20.describe())
    data20 = wu20.generate(n_samples=200_000)
    cs20   = wu20.extract_constraints(data20)
    X_size_20 = int(np.prod(wu20.domain_sizes))
    print(f"\n{cs20.summary()}")
    print(f"|X| = {X_size_20:,}  "
          f"({'enumerable' if X_size_20 < 5e6 else 'not enumerable'})")

    print("\n--- GibbsPCDSolver ---")
    a2b_rows = []
    configs = [
        (50_000,  3, 0.01),
        (100_000, 3, 0.01),
        (100_000, 5, 0.01),
    ]
    for N, s, lr in configs:
        r = run_gibbs_a2(cs20, N_pool=N, n_gibbs_sweeps=s, lr=lr,
                         n_outer=600, tol=0.02, window=50,
                         seed=1, label=f'K=20 N={N//1000}K s={s}',
                         use_numba=use_numba)
        r['KL'] = r['pdist'] = float('nan')
        pool20 = r['solver']._init_pool(N, seed=99)
        for _ in range(100):
            pool20 = r['solver']._gibbs_sweep(pool20, r['solver'].lambdas)
        r['neff'] = N
        r['n_unique'], r['H'] = pool_diversity(pool20)
        a2b_rows.append(r)
        print(f"  {r['label']}: MRE={r['MRE']:.5f}  t={r['fit_time']:.1f}s")

    print("\n--- Raking ---")
    for N in [50_000, 100_000]:
        r = run_raking(cs20, N_pop=N, tol=0.02, window=30,
                       seed=1, label=f'K=20 Raking N={N//1000}K')
        neff_r, n_unique_r, H_r = raking_diversity(r['solver'])
        r['neff']     = neff_r
        r['n_unique'] = n_unique_r
        r['H']        = H_r
        a2b_rows.append(r)
        print(f"  {r['label']}: MRE={r['MRE']:.5f}  "
              f"N_eff={neff_r:.0f}  t={r['fit_time']:.1f}s")

    print_a2_table(a2b_rows, 'A2b — K=20 (ternary, MCMC)')
    return a2b_rows


# ------------------------------------------------------------------ #
#  A2c — K=30                                                          #
# ------------------------------------------------------------------ #

def run_a2c(use_numba=True):
    print("\n" + "="*70)
    print("  A2c — K=30  (pure MCMC, beyond Pachet-Zucker boundary)")
    print("="*70)

    wu30 = WuGenerator(K=30, domain_range=(2, 3), n_patterns=15,
                       pattern_arity=3, seed=40)
    print(wu30.describe())
    data30 = wu30.generate(n_samples=200_000)
    cs30   = wu30.extract_constraints(data30)
    X_size_30 = int(np.prod(wu30.domain_sizes))
    print(f"\n{cs30.summary()}")
    print(f"|X| = {X_size_30:,}  (not enumerable by construction)")

    print("\n--- GibbsPCDSolver (Numba recommended) ---")
    a2c_rows = []
    configs = [
        (50_000,  5, 0.01),
        (100_000, 5, 0.01),
    ]
    for N, s, lr in configs:
        r = run_gibbs_a2(cs30, N_pool=N, n_gibbs_sweeps=s, lr=lr,
                         n_outer=600, tol=0.02, window=50,
                         seed=1, label=f'K=30 N={N//1000}K s={s}',
                         use_numba=use_numba)
        r['KL'] = r['pdist'] = float('nan')
        pool30 = r['solver']._init_pool(N, seed=99)
        for _ in range(100):
            pool30 = r['solver']._gibbs_sweep(pool30, r['solver'].lambdas)
        r['neff'] = N
        r['n_unique'], r['H'] = pool_diversity(pool30)
        a2c_rows.append(r)
        print(f"  {r['label']}: MRE={r['MRE']:.5f}  t={r['fit_time']:.1f}s")

    print("\n--- Raking ---")
    for N in [100_000]:
        r = run_raking(cs30, N_pop=N, tol=0.02, window=30,
                       seed=1, label=f'K=30 Raking N={N//1000}K')
        neff_r, n_unique_r, H_r = raking_diversity(r['solver'])
        r['neff']     = neff_r
        r['n_unique'] = n_unique_r
        r['H']        = H_r
        a2c_rows.append(r)
        print(f"  {r['label']}: MRE={r['MRE']:.5f}  "
              f"N_eff={neff_r:.0f}  t={r['fit_time']:.1f}s")

    print_a2_table(a2c_rows, 'A2c — K=30 (ternary, pure MCMC)')
    return a2c_rows


# ------------------------------------------------------------------ #
#  Main                                                                 #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment A2 — scaling K=12..30')
    parser.add_argument('--no_numba', action='store_true',
                        help='Disable Numba acceleration (pure NumPy)')
    parser.add_argument('--skip_a2c', action='store_true',
                        help='Skip K=30 experiment (fastest run)')
    args = parser.parse_args()

    use_numba = not args.no_numba

    print("=" * 70)
    print("EXPERIMENT A2: Scaling beyond exact enumeration")
    print(f"  Numba: {'enabled' if use_numba else 'disabled'}")
    print("=" * 70)

    rows_a2a = run_a2a(use_numba=False)   # K=12: Numba not needed
    rows_a2b = run_a2b(use_numba=use_numba)
    if not args.skip_a2c:
        rows_a2c = run_a2c(use_numba=use_numba)

    print("\n\nAll A2 experiments complete.")
