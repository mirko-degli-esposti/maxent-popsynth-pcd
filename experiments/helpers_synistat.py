"""
helpers_synistat.py
-------------------
Wrappers and helpers for Syn-ISTAT experiments (A-ISTAT-1..3, DIV).
"""

import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import GibbsPCDSolver


def run_gibbs_std(cs, N_pool, n_gibbs_sweeps=5,
                  tol=0.02, window=50, lr=0.01,
                  seed=1, label='', verbose_every=50):
    """Standard GibbsPCDSolver run for Syn-ISTAT experiments."""
    g = GibbsPCDSolver(cs, use_numba=True)
    g.fit(N_pool=N_pool, n_outer=600,
          n_gibbs_sweeps=n_gibbs_sweeps,
          lr=lr, tol=tol, window=window,
          seed=seed, verbose_every=verbose_every)
    return {
        'label':         label,
        'N_pool':        N_pool,
        'n_sweeps':      n_gibbs_sweeps,
        'MRE':           g.final_mre,
        'n_iters':       g.n_iters,
        'stopped_early': g.stopped_early,
        'fit_time':      g.fit_time,
        'solver':        g,
        'method':        'gibbs',
    }


def run_raking(cs, N_pop, max_iter=1000, tol=0.02,
               window=30, seed=1, label=''):
    """Raking on a uniform synthetic population."""
    from experiments.helpers_a2 import _RakingSolver
    rng = np.random.default_rng(seed)
    population = np.zeros((N_pop, cs.K), dtype=np.int32)
    for k in range(cs.K):
        population[:, k] = rng.integers(0, cs.domain_sizes[k], size=N_pop)

    r = _RakingSolver(cs)
    r.fit(population, max_iter=max_iter, tol=tol, window=window)

    weights_norm = r.weights / r.weights.sum()
    alpha_hat = np.array([
        float(weights_norm[r._masks[j]].sum()) for j in range(cs.m)
    ])
    mre_weighted = float(np.mean(
        np.abs(alpha_hat - cs.alphas_array) / (cs.alphas_array + 1e-12)
    ))

    return {
        'label':         label,
        'N_pool':        N_pop,
        'n_sweeps':      0,
        'MRE':           mre_weighted,
        'MRE_train':     r.final_mre,
        'n_iters':       r.n_iters,
        'stopped_early': r.n_iters < max_iter,
        'fit_time':      r.fit_time,
        'solver':        r,
        'method':        'raking',
    }


def mre_on_cs_ext(solver_result, cs_eval):
    """
    Compute MRE on cs_eval (different from training cs).

    For Gibbs: uses solver.pool_ (final persistent pool).
    For Raking: uses solver.weights and ._population.

    Returns
    -------
    mre : float
    alpha_hat : (cs_eval.m,) array of estimated frequencies
    """
    s      = solver_result['solver']
    method = solver_result['method']

    if method == 'gibbs':
        pop = s.pool_
        wts = np.ones(len(pop)) / len(pop)
    else:
        pop = s._population
        wts = s.weights / s.weights.sum()

    alpha_hat = np.zeros(cs_eval.m)
    for j in range(cs_eval.m):
        attrs = cs_eval.attrs_list[j]
        vals  = cs_eval.vals_list[j]
        mask  = np.all(pop[:, attrs] == vals[np.newaxis, :], axis=1)
        alpha_hat[j] = float(wts[mask].sum())

    mre = float(np.mean(
        np.abs(alpha_hat - cs_eval.alphas_array) /
        (cs_eval.alphas_array + 1e-12)
    ))
    return mre, alpha_hat


def print_comparison(rows, title=''):
    """Print comparison table: label, method, N, sweeps, MRE, iters, time."""
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    print(f"  {'Label':<22} {'Method':<8} {'N':>7} {'s':>3} "
          f"{'MRE':>8} {'Iters':>6} {'t(s)':>7}")
    print(f"  {'-'*72}")
    for r in rows:
        print(f"  {r['label']:<22} {r['method']:<8} {r['N_pool']:>7,} "
              f"{r['n_sweeps']:>3} {r['MRE']:>8.5f} "
              f"{r['n_iters']:>6} {r['fit_time']:>7.1f}")
    print(f"{'='*75}")


def empirical_entropy(rows, weights=None):
    """
    Shannon entropy H = -sum p(x) log p(x) over unique profiles.

    Parameters
    ----------
    rows : list of tuples (demographic profiles)
    weights : None -> uniform (Gibbs), array -> raking weights
    """
    from collections import defaultdict
    counts = defaultdict(float)
    if weights is None:
        for r in rows:
            counts[r] += 1.0 / len(rows)
    else:
        for r, w in zip(rows, weights):
            counts[r] += float(w)
    probs = np.array(list(counts.values()))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def gini(w):
    """Gini coefficient: 0 = uniform, 1 = fully concentrated."""
    w = np.sort(np.array(w, dtype=np.float64))
    n = len(w)
    cumw = np.cumsum(w)
    return float(1 - 2 * np.sum(cumw) / (n * cumw[-1]) + 1 / n)
