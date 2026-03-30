"""
run_AISTAT_heldout.py
---------------------
Experiment A-ISTAT-2: Held-out generalisation on Syn-ISTAT (K=15).

Both solvers are trained on 28 unary+binary constraints (cs_train28).
The 3 ternary constraints (T1, T2, T3) are withheld and evaluated
after training to test generalisation to unseen higher-order interactions.

Key findings (paper Table 3):
    - GibbsPCDSolver MRE_train = 0.018 (600 iters)
    - Raking MRE_train = 0.000 (exact by construction)
    - GibbsPCDSolver outperforms raking on T3 (sex x age -> employment)
      the highest-degree constraint (degree 5 in the constraint graph)
    - Raking outperforms on T2 (anchor-only conditioning variables)

Output:
    Console table (no figures)

Usage:
    python experiments/run_AISTAT_heldout.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.syn_istat import build_syn_istat_constraint_sets, ATTR_META
from experiments.helpers_synistat import (
    run_gibbs_std, run_raking, mre_on_cs_ext, print_comparison
)


def run_aistat_heldout():
    print("=" * 70)
    print("  A-ISTAT-2: Held-out evaluation (train 28, test 3 ternary)")
    print("=" * 70)

    cs_full, cs_train28, cs_held3 = build_syn_istat_constraint_sets()

    N = 100_000

    # ── Training ─────────────────────────────────────────────────
    print("\n--- GibbsPCDSolver (training on 28 unary+binary) ---")
    res_g2 = run_gibbs_std(cs_train28, N_pool=N, seed=42,
                            label='K=15 Gibbs', verbose_every=100)

    print("\n--- Raking (training on 28 unary+binary) ---")
    res_r2 = run_raking(cs_train28, N_pop=N,
                         tol=0.02, window=30, seed=42,
                         label='K=15 Raking')

    print_comparison([res_g2, res_r2],
                     title='A-ISTAT-2: Training MRE (28 unary+binary)')

    # ── Held-out evaluation ───────────────────────────────────────
    mre_held_g, alpha_held_g = mre_on_cs_ext(res_g2, cs_held3)
    mre_held_r, alpha_held_r = mre_on_cs_ext(res_r2, cs_held3)

    print(f"\n  {'':22} {'Method':<8}  {'MRE held-out (3 ternary)':>24}")
    print(f"  {'-'*58}")
    print(f"  {'K=15 Gibbs':<22} {'gibbs':<8}  {mre_held_g:>24.5f}")
    print(f"  {'K=15 Raking':<22} {'raking':<8}  {mre_held_r:>24.5f}")
    print(f"\n  Held-out ratio MRE_raking/MRE_gibbs = "
          f"{mre_held_r / max(mre_held_g, 1e-9):.2f}x  "
          f"({'Gibbs generalises better' if mre_held_g < mre_held_r else 'Raking generalises better'})")

    # ── Per-ternary breakdown ─────────────────────────────────────
    ternary_keys = [
        ('education', 'employment', 'income'),
        ('residence_area', 'car_access', 'main_transport'),
        ('sex', 'age', 'employment'),
    ]
    ternary_labels = ['T1: edu x emp -> income',
                      'T2: area x car -> transport',
                      'T3: sex x age -> employment']

    print(f"\n  Per-ternary held-out MRE:")
    print(f"  {'Ternary':<48}  {'Gibbs':>8}  {'Raking':>8}")
    print(f"  {'-'*68}")

    idx = 0
    for tkey, tlabel in zip(ternary_keys, ternary_labels):
        n_cells = int(np.prod([len(ATTR_META[n]['vals']) for n in tkey]))
        tgt   = cs_held3.alphas_array[idx:idx + n_cells]
        a_g   = alpha_held_g[idx:idx + n_cells]
        a_r   = alpha_held_r[idx:idx + n_cells]
        mre_g = float(np.mean(np.abs(a_g - tgt) / (tgt + 1e-12)))
        mre_r = float(np.mean(np.abs(a_r - tgt) / (tgt + 1e-12)))
        winner = '<- Gibbs' if mre_g < mre_r else '<- Raking'
        print(f"  {tlabel:<48}  {mre_g:>8.4f}  {mre_r:>8.4f}  {winner}")
        idx += n_cells

    return res_g2, res_r2, cs_held3


if __name__ == '__main__':
    run_aistat_heldout()
