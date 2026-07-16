"""
gibbs_pcd_solver.py
-------------------
GibbsPCDSolver: scalable Maximum Entropy population synthesis
via Persistent Contrastive Divergence.

Replaces the exact expectation step of Pachet & Zucker (2026)
with a stochastic approximation from a persistent Gibbs pool,
removing the |X| barrier that limits exact MaxEnt to K ~< 20.

Reference:
    Degli Esposti, M. (2026). Scalable Maximum Entropy Population
    Synthesis via Persistent Contrastive Divergence. arXiv:2503.XXXXX
    Pachet & Zucker (2026). Maximum Entropy Relaxation of Multi-Way
    Cardinality Constraints for Synthetic Population Generation.
    Tieleman, T. (2008). Training Restricted Boltzmann Machines Using
    Approximations to the Likelihood Gradient. ICML.
"""

import time
import numpy as np
from .constraint_set import ConstraintSet


class GibbsPCDSolver:
    """
    Maximum Entropy solver via Persistent Contrastive Divergence.

    Maintains a persistent pool of N synthetic individuals updated
    by Gibbs (Glauber) sweeps at each gradient step. The pool's
    empirical frequencies provide a stochastic approximation of
    the model expectations E_{p_lambda}[f_j], without ever
    materialising X or computing Z(lambda).

    Parameters
    ----------
    cs : ConstraintSet
        Constraint set defining the MaxEnt problem.
    use_numba : bool, optional
        If True, use the Numba-accelerated Gibbs kernel (recommended
        for K >= 20). Falls back to pure NumPy if Numba is unavailable.

    Notes
    -----
    The Gibbs conditional update for attribute k is:

        p(A_k = v | x_{-k}) ∝ exp( sum_{j in J(k,v,x_{-k})} lambda_j )

    where J(k, v, x_{-k}) is the set of constraints j with k in S_j,
    v_j^(k) = v, and x_{S_j \\ {k}} = v_j^(-k).

    This leaves p_lambda invariant by detailed balance, and the chain
    is ergodic because p_lambda > 0 on all of X.

    Attributes updated by fit()
    ---------------------------
    lambdas : (m,) float64 array — learned Lagrange multipliers
    history : list of dicts — per-iteration diagnostics
    final_mre : float — MRE at convergence
    fit_time : float — wall-clock seconds
    n_iters : int — actual iterations run
    stopped_early : bool
    """

    def __init__(self, cs: ConstraintSet, use_numba: bool = False):
        self.cs       = cs
        self.K        = cs.K
        self.m        = cs.m
        self.alphas   = cs.alphas_array
        self.use_numba = use_numba
        self.pool_: np.ndarray | None = None

        # Precompute attr_lookup for vectorised energy accumulation
        self.lookup = cs.build_attr_lookup()

        # Numba kernel + flattened CSR lookup (dicts are not Numba-typable)
        self._numba_kernel = None
        self._nb_lookup    = None
        if use_numba:
            self._numba_kernel = _make_gibbs_numba_kernel()
            if self._numba_kernel is not None:
                self._nb_lookup = self._build_numba_lookup()
                self._nb_domain_sizes = np.asarray(cs.domain_sizes,
                                                   dtype=np.int64)

        # Results (populated by fit)
        self.lambdas:      np.ndarray | None = None
        self.history:      list[dict]        = []
        self.final_mre:    float             = float('nan')
        self.fit_time:     float             = 0.0
        self.n_iters:      int               = 0
        self.stopped_early: bool             = False

    # ------------------------------------------------------------------ #
    #  Pool initialisation                                                  #
    # ------------------------------------------------------------------ #

    def _init_pool(self, N: int, seed: int = 1) -> np.ndarray:
        """Initialise pool uniformly over attribute domains."""
        rng  = np.random.default_rng(seed)
        pool = np.zeros((N, self.K), dtype=np.int32)
        for k in range(self.K):
            pool[:, k] = rng.integers(0, self.cs.domain_sizes[k], size=N)
        return pool

    # ------------------------------------------------------------------ #
    #  Gibbs sweep (pure NumPy)                                            #
    # ------------------------------------------------------------------ #

    def _gibbs_sweep(self, pool: np.ndarray,
                     lam: np.ndarray) -> np.ndarray:
        """
        One full Gibbs sweep over all K attributes (random permuted order).

        For each attribute k, computes log-energies of shape (N, d_k)
        via the precomputed lookup table, applies numerically stable
        softmax, and samples new values for all N individuals.

        Cost: O(N * K * d_max * mean_J) per sweep.
        """
        N   = pool.shape[0]
        rng = np.random.default_rng()

        for k in rng.permutation(self.K):
            d_k     = int(self.cs.domain_sizes[k])
            log_e   = np.zeros((N, d_k), dtype=np.float64)

            for (j, v_k, other_attrs, other_vals) in self.lookup[k]:
                # Identify individuals whose context matches this constraint
                if len(other_attrs) > 0:
                    ctx_match = np.all(
                        pool[:, other_attrs] == other_vals[np.newaxis, :],
                        axis=1
                    )
                else:
                    ctx_match = np.ones(N, dtype=bool)
                log_e[ctx_match, v_k] += lam[j]

            # Numerically stable softmax -> categorical sample
            log_e -= log_e.max(axis=1, keepdims=True)
            probs  = np.exp(log_e)
            probs /= probs.sum(axis=1, keepdims=True)

            # Vectorised categorical sampling for all N individuals
            cdf  = probs.cumsum(axis=1)
            u    = np.random.rand(N, 1)
            pool[:, k] = (u > cdf).sum(axis=1).clip(0, d_k - 1)

        return pool

    # ------------------------------------------------------------------ #
    #  Expectation estimation                                               #
    # ------------------------------------------------------------------ #

    def _estimate_expectations(self, pool: np.ndarray) -> np.ndarray:
        """
        Estimate alpha_hat_j = (1/N) sum_i f_j(pool[i]) for all j.
        """
        alpha_hat = np.zeros(self.m, dtype=np.float64)
        for j in range(self.m):
            attrs = self.cs.attrs_list[j]
            vals  = self.cs.vals_list[j]
            alpha_hat[j] = np.all(
                pool[:, attrs] == vals[np.newaxis, :], axis=1
            ).mean()
        return alpha_hat

    # ------------------------------------------------------------------ #
    #  Main fitting loop                                                    #
    # ------------------------------------------------------------------ #

    def fit(self,
            N_pool:         int   = 10_000,
            n_outer:        int   = 500,
            n_gibbs_sweeps: int   = 5,
            lr:             float = 0.01,
            beta1:          float = 0.9,
            beta2:          float = 0.999,
            eps:            float = 1e-8,
            seed:           int   = 1,
            tol:            float = 0.02,
            window:         int   = 50,
            verbose_every:  int   = 50) -> 'GibbsPCDSolver':
        """
        Fit GibbsPCDSolver using Adam optimiser with adaptive stopping.

        Parameters
        ----------
        N_pool : int
            Persistent pool size. MRE floor ~ 1/(2*sqrt(N_pool)).
            Recommended: 25_000-100_000. Sweet spot at 25_000 for K=15.
        n_outer : int
            Maximum number of outer (gradient) iterations.
        n_gibbs_sweeps : int
            Number of Gibbs sweeps per gradient step.
            Use s=1 for sparse binary constraints (K<=12),
            s=5 for mixed arity (K=15), s=5 for K>=20.
        lr : float
            Adam learning rate. Use 0.01 for K>=15 (0.05 oscillates).
        tol : float
            Adaptive stopping threshold (relative MRE improvement).
            Set tol=0.0 to disable early stopping.
        window : int
            Window length for adaptive stopping rule.
            Stopping fires when relative improvement of min(MRE) over
            two consecutive windows falls below tol.
        verbose_every : int
            Print progress every this many iterations (0 = silent).

        Returns
        -------
        self : GibbsPCDSolver (fitted)
        """
        lam  = np.zeros(self.m, dtype=np.float64)
        pool = self._init_pool(N_pool, seed=seed)

        sweep_fn = (self._gibbs_sweep_numba
                    if (self.use_numba and self._numba_kernel is not None)
                    else self._gibbs_sweep)

        # Adam state
        m1 = np.zeros(self.m, dtype=np.float64)
        m2 = np.zeros(self.m, dtype=np.float64)

        self.history       = []
        self.stopped_early = False
        t_start            = time.time()

        for t in range(1, n_outer + 1):

            # Inner loop: Gibbs sweeps on persistent pool
            for _ in range(n_gibbs_sweeps):
                pool = sweep_fn(pool, lam)

            # Stochastic gradient estimate: grad = alpha_hat - alpha
            alpha_hat = self._estimate_expectations(pool)
            grad      = alpha_hat - self.alphas

            # Adam update
            m1  = beta1 * m1 + (1.0 - beta1) * grad
            m2  = beta2 * m2 + (1.0 - beta2) * grad ** 2
            m1h = m1 / (1.0 - beta1 ** t)
            m2h = m2 / (1.0 - beta2 ** t)
            lam -= lr * m1h / (np.sqrt(m2h) + eps)

            mre = float(np.mean(
                np.abs(alpha_hat - self.alphas) / (self.alphas + 1e-12)
            ))

            self.history.append({
                't':         t,
                'mre':       mre,
                'alpha_hat': alpha_hat.copy(),
                'elapsed':   time.time() - t_start,
            })

            if verbose_every and t % verbose_every == 0:
                print(f"  [Gibbs] iter {t:4d}  MRE={mre:.5f}  "
                      f"N={N_pool}  t={time.time()-t_start:.1f}s")

            # Adaptive stopping rule (Eq. 7 in paper)
            # Stops when relative improvement of min(MRE) over two
            # consecutive windows of length `window` falls below `tol`.
            if tol > 0.0 and t >= 2 * window:
                recent_min  = min(h['mre'] for h in self.history[-window:])
                earlier_min = min(h['mre'] for h in
                                  self.history[-2*window:-window])
                if earlier_min > 0:
                    rel_improv = (earlier_min - recent_min) / earlier_min
                else:
                    rel_improv = 0.0
                if rel_improv < tol:
                    self.stopped_early = True
                    if verbose_every:
                        print(f"  [Gibbs] Early stop at iter {t}  "
                              f"rel_improv={rel_improv:.4f} < tol={tol}  "
                              f"MRE={mre:.5f}")
                    break

        self.lambdas   = lam
        self.pool_     = pool          # persistent pool at convergence
        self.fit_time  = time.time() - t_start
        self.final_mre = self.history[-1]['mre']
        self.n_iters   = len(self.history)
        return self

    def _gibbs_sweep_numba(self, pool: np.ndarray,
                           lam: np.ndarray) -> np.ndarray:
        """Numba-accelerated Gibbs sweep (falls back to NumPy if unavailable)."""
        if self._numba_kernel is None or self._nb_lookup is None:
            return self._gibbs_sweep(pool, lam)
        es, ee, ej, ev, cp, ca, cv = self._nb_lookup
        return self._numba_kernel(pool, lam, self._nb_domain_sizes,
                                  es, ee, ej, ev, cp, ca, cv, self.K)

    def _build_numba_lookup(self):
        """Flatten build_attr_lookup() into typed CSR arrays for Numba.

        Returns (ent_start, ent_end, ent_j, ent_vk,
                 ctx_ptr, ctx_attrs, ctx_vals):
        entries are concatenated in attribute order; for attribute k the
        entries live in [ent_start[k], ent_end[k]); entry e constrains
        value ent_vk[e] of attribute k with multiplier lambda[ent_j[e]],
        conditional on pool[i, ctx_attrs[p]] == ctx_vals[p] for
        p in [ctx_ptr[e], ctx_ptr[e+1]).
        """
        ent_start = np.zeros(self.K, dtype=np.int64)
        ent_end   = np.zeros(self.K, dtype=np.int64)
        ent_j, ent_vk = [], []
        ctx_ptr, ctx_attrs, ctx_vals = [0], [], []
        e = 0
        for k in range(self.K):
            ent_start[k] = e
            for (j, v_k, oa, ov) in self.lookup.get(k, []):
                ent_j.append(j); ent_vk.append(v_k)
                ctx_attrs.extend(np.asarray(oa).tolist())
                ctx_vals.extend(np.asarray(ov).tolist())
                ctx_ptr.append(len(ctx_attrs))
                e += 1
            ent_end[k] = e
        return (ent_start, ent_end,
                np.asarray(ent_j,     dtype=np.int64),
                np.asarray(ent_vk,    dtype=np.int64),
                np.asarray(ctx_ptr,   dtype=np.int64),
                np.asarray(ctx_attrs, dtype=np.int64),
                np.asarray(ctx_vals,  dtype=np.int64))

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def mre_curve(self) -> np.ndarray:
        """Return MRE values across iterations as a 1D array."""
        return np.array([h['mre'] for h in self.history])

    def __repr__(self):
        status = (f"fitted, MRE={self.final_mre:.4f}, "
                  f"{self.n_iters} iters"
                  if self.lambdas is not None else "not fitted")
        return f"GibbsPCDSolver(K={self.K}, m={self.m}, {status})"


# ------------------------------------------------------------------ #
#  Optional Numba kernel (standalone, outside class)                   #
# ------------------------------------------------------------------ #

def _make_gibbs_numba_kernel():
    """
    Build and return a Numba-jitted Gibbs sweep kernel.

    Operates on the flattened CSR lookup produced by
    GibbsPCDSolver._build_numba_lookup() (plain int64 arrays), since
    Numba nopython mode cannot type the dict-of-tuples attr lookup.
    Parallelised over individuals with prange. Returns None if Numba
    is not installed.
    """
    try:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _gibbs_sweep_numba_kernel(pool, lam, domain_sizes,
                                      ent_start, ent_end, ent_j, ent_vk,
                                      ctx_ptr, ctx_attrs, ctx_vals, K):
            N = pool.shape[0]
            attr_order = np.arange(K)
            np.random.shuffle(attr_order)

            for ki in range(K):
                k   = attr_order[ki]
                d_k = domain_sizes[k]
                e0  = ent_start[k]
                e1  = ent_end[k]

                for i in prange(N):
                    log_e = np.zeros(d_k)
                    for e in range(e0, e1):
                        match = True
                        for p in range(ctx_ptr[e], ctx_ptr[e + 1]):
                            if pool[i, ctx_attrs[p]] != ctx_vals[p]:
                                match = False
                                break
                        if match:
                            log_e[ent_vk[e]] += lam[ent_j[e]]

                    # stable softmax + categorical sample
                    mx = log_e[0]
                    for v in range(1, d_k):
                        if log_e[v] > mx:
                            mx = log_e[v]
                    s = 0.0
                    for v in range(d_k):
                        log_e[v] = np.exp(log_e[v] - mx)
                        s += log_e[v]
                    u = np.random.random() * s
                    csum = 0.0
                    chosen = d_k - 1
                    for v in range(d_k):
                        csum += log_e[v]
                        if u <= csum:
                            chosen = v
                            break
                    pool[i, k] = chosen

            return pool

        return _gibbs_sweep_numba_kernel

    except ImportError:
        return None
