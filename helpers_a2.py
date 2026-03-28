"""
helpers_a2.py
-------------
Helper functions and wrappers for A2 scaling experiments.

Imported by run_A2_*.py scripts. Not part of the core src/ library.
"""

import time
from collections import Counter
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import GibbsPCDSolver


# ------------------------------------------------------------------ #
#  Gibbs wrapper                                                        #
# ------------------------------------------------------------------ #

def run_gibbs_a2(cs, N_pool, n_gibbs_sweeps, lr, n_outer=600,
                 tol=0.02, window=50, seed=1, label='',
                 use_numba=False):
    """
    GibbsPCDSolver wrapper for A2 experiments.

    Parameters passed explicitly (no fixed defaults) to allow
    per-K calibration as described in the paper.

    Returns
    -------
    dict with keys: label, N_pool, n_sweeps, lr, MRE, n_iters,
                    stopped_early, fit_time, solver, method
    """
    g = GibbsPCDSolver(cs, use_numba=use_numba)
    g.fit(N_pool=N_pool, n_outer=n_outer,
          n_gibbs_sweeps=n_gibbs_sweeps,
          lr=lr, tol=tol, window=window,
          seed=seed, verbose_every=0)
    return {
        'label':         label,
        'N_pool':        N_pool,
        'n_sweeps':      n_gibbs_sweeps,
        'lr':            lr,
        'MRE':           g.final_mre,
        'n_iters':       g.n_iters,
        'stopped_early': g.stopped_early,
        'fit_time':      g.fit_time,
        'solver':        g,
        'method':        'gibbs',
    }


# ------------------------------------------------------------------ #
#  Raking wrapper                                                       #
# ------------------------------------------------------------------ #

def run_raking(cs, N_pop, max_iter=1000, tol=0.02, window=30,
               seed=1, label=''):
    """
    Raking on a uniform synthetic population of N_pop individuals.

    MRE is computed from weighted frequencies (directly comparable
    to GibbsPCDSolver MRE). The population and masks are saved on
    the solver object for downstream diversity metrics.

    Returns
    -------
    dict with keys: label, N_pool, n_sweeps, lr, MRE, MRE_train,
                    n_iters, stopped_early, fit_time, solver, method
    """
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
        'lr':            0.0,
        'MRE':           mre_weighted,
        'MRE_train':     r.final_mre,
        'n_iters':       r.n_iters,
        'stopped_early': r.n_iters < max_iter,
        'fit_time':      r.fit_time,
        'solver':        r,
        'method':        'raking',
    }


# ------------------------------------------------------------------ #
#  Diversity metrics                                                    #
# ------------------------------------------------------------------ #

def neff(weights):
    """Effective sample size: N_eff = 1 / sum(w^2)."""
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def pool_diversity(pool):
    """
    Diversity of a Gibbs pool (uniform weights by construction).

    Parameters
    ----------
    pool : (N, K) int32 array — persistent Gibbs pool

    Returns
    -------
    n_unique : int — number of distinct profiles
    H : float — empirical Shannon entropy in nats
    """
    rows    = [tuple(pool[i]) for i in range(len(pool))]
    counts  = Counter(rows)
    n_unique = len(counts)
    probs   = np.array(list(counts.values()), dtype=np.float64)
    probs  /= probs.sum()
    H       = float(-np.sum(probs * np.log(probs + 1e-300)))
    return n_unique, H


def raking_diversity(solver_r):
    """
    Compute N_eff, unique profiles, and weighted entropy for a
    fitted _RakingSolver.

    Returns
    -------
    neff_r : float
    n_unique_r : int
    H_r : float — weighted Shannon entropy in nats
    """
    pop = solver_r._population
    w   = solver_r.weights / solver_r.weights.sum()

    neff_r = float(1.0 / np.sum(w ** 2))

    # Weighted probability per unique profile
    rows = [tuple(pop[i]) for i in range(len(pop))]
    wts_dict = {}
    for i, r in enumerate(rows):
        wts_dict[r] = wts_dict.get(r, 0.0) + float(w[i])

    probs     = np.array(list(wts_dict.values()))
    H_r       = float(-np.sum(probs * np.log(probs + 1e-300)))
    n_unique_r = len(wts_dict)
    return neff_r, n_unique_r, H_r


# ------------------------------------------------------------------ #
#  Pretty-print                                                         #
# ------------------------------------------------------------------ #

def print_a2_table(rows, title=''):
    """Print A2 results table: MRE, time, N_eff, unique profiles, H."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Label':<22} {'Method':<8} {'N':>7} {'s':>3} {'lr':>6} "
          f"{'MRE':>8} {'t(s)':>7} {'N_eff':>9} {'Uniq':>8} {'H':>7}")
    print(f"  {'-'*77}")
    for r in rows:
        print(f"  {r['label']:<22} {r.get('method','?'):<8} "
              f"{r['N_pool']:>7,} {r['n_sweeps']:>3} {r.get('lr',0):>6.3f} "
              f"{r['MRE']:>8.5f} {r['fit_time']:>7.1f} "
              f"{r.get('neff', float('nan')):>9.0f} "
              f"{r.get('n_unique', 0):>8,} "
              f"{r.get('H', float('nan')):>7.3f}")
    print(f"{'='*80}")


# ------------------------------------------------------------------ #
#  Local RakingSolver (A2 variant with precomputed masks)              #
# ------------------------------------------------------------------ #

class _RakingSolver:
    """
    Raking solver for A2 experiments.

    Extends the core RakingSolver with precomputed boolean masks
    (for fast repeated weight accumulation) and stores the population
    array for downstream diversity metrics.

    Not part of the public src/ API — use src.RakingSolver for
    general purposes.
    """

    def __init__(self, cs):
        self.cs        = cs
        self.alphas    = cs.alphas_array
        self.m         = cs.m
        self.weights   = None
        self.history   = []
        self.fit_time  = 0.0
        self.final_mre = float('nan')
        self.n_iters   = 0
        self._population = None
        self._masks      = None

    def fit(self, population, max_iter=1000, tol=0.02,
            window=30, verbose_every=0):
        N_pop   = len(population)
        weights = np.ones(N_pop, dtype=np.float64) / N_pop

        # Precompute boolean masks for each constraint
        masks = []
        for j in range(self.m):
            attrs = self.cs.attrs_list[j]
            vals  = self.cs.vals_list[j]
            masks.append(np.all(
                population[:, attrs] == vals[np.newaxis, :], axis=1
            ))
        self._masks      = masks
        self._population = population

        self.history = []
        t_start = time.time()

        for t in range(1, max_iter + 1):
            for j in range(self.m):
                f_hat = float(weights[masks[j]].sum())
                if f_hat > 1e-15:
                    weights[masks[j]] *= self.alphas[j] / f_hat
                w_sum = weights.sum()
                if w_sum > 0:
                    weights /= w_sum

            alpha_hat = np.array([
                float(weights[masks[j]].sum()) for j in range(self.m)
            ])
            mre = float(np.mean(
                np.abs(alpha_hat - self.alphas) / (self.alphas + 1e-12)
            ))
            self.history.append({'t': t, 'mre': mre})

            if verbose_every and t % verbose_every == 0:
                print(f"  [Raking] iter {t:4d}  MRE={mre:.5f}")

            # Adaptive stopping (same rule as GibbsPCDSolver)
            if tol > 0.0 and t >= 2 * window:
                recent_min  = min(h['mre'] for h in self.history[-window:])
                earlier_min = min(h['mre'] for h in
                                  self.history[-2*window:-window])
                rel_improv  = ((earlier_min - recent_min) / earlier_min
                               if earlier_min > 0 else 0.0)
                if rel_improv < tol:
                    break

        self.weights   = weights
        self.fit_time  = time.time() - t_start
        self.final_mre = self.history[-1]['mre']
        self.n_iters   = len(self.history)
        return self
