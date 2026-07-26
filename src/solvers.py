"""
solvers.py
----------
Reference solvers for MaxEnt population synthesis.

Classes
-------
ExactMaxEntSolver
    L-BFGS solver with exact expectation computation by enumeration.
    Reference implementation of Pachet & Zucker (2026) Algorithm 1.
    Feasible for |X| <= ~10^7 (K <= ~20 with binary domains).

RakingSolver
    Generalised raking (Iterative Proportional Fitting, IPF).
    Fast baseline that reweights a fixed sample to match aggregate
    constraints. Exact on training by algebraic construction, but
    produces degenerate populations (N_eff << N) at large K.
"""

import time
import numpy as np
from itertools import product as cart_product
from scipy.special import logsumexp
from scipy.optimize import minimize
from .constraint_set import ConstraintSet


class ExactMaxEntSolver:
    """
    Exact MaxEnt solver via L-BFGS + full enumeration of X.

    Solves: min_lambda Phi(lambda) = log Z(lambda) - lambda . alpha
    where Z(lambda) = sum_{x in X} exp( sum_j lambda_j f_j(x) )

    Complexity: O(|X| * m) per iteration.
    Practical limit: |X| <= ~10^7.

    Parameters
    ----------
    cs : ConstraintSet
    verbose : bool
    """

    def __init__(self, cs: ConstraintSet, verbose: bool = True,
                 sparse: bool = False, F=None, all_tuples=None):
        self.cs     = cs
        self.K      = cs.K
        self.alphas = cs.alphas_array
        self.sparse = sparse

        X_size = int(np.prod(cs.domain_sizes))
        self.X_size = X_size
        if verbose:
            print(f"  [Exact] Enumeration: |X| = {X_size:,}, m = {cs.m}")

        # all_tuples serve SOLO a costruire F: se F arriva dall'esterno, non serve.
        if all_tuples is not None:
            self.all_tuples = all_tuples
        elif F is None:
            g = np.meshgrid(*[np.arange(d) for d in cs.domain_sizes], indexing="ij")
            self.all_tuples = np.stack([x.ravel() for x in g],
                                       axis=1).astype(np.int32)
        else:
            self.all_tuples = None

        t0 = time.time()
        if F is not None:
            if F.shape != (X_size, cs.m):
                raise ValueError(f"F fornita ha shape {F.shape}, "
                                 f"attesa ({X_size}, {cs.m})")
            self.F = F
            if verbose:
                print(f"  [Exact] F riusata dall'esterno (nessuna ricostruzione)")
        else:
            if sparse:
                self.F = cs.build_indicator_matrix_sparse(self.all_tuples)
            else:
                self.F = cs.build_indicator_matrix(self.all_tuples)
            if verbose:
                nbytes = (self.F.data.nbytes + self.F.indices.nbytes
                          + self.F.indptr.nbytes) if sparse else self.F.nbytes
                print(f"  [Exact] F built in {time.time()-t0:.2f}s  "
                      f"({nbytes / 1e6:.1f} MB)")

        self.lambdas:   np.ndarray | None = None
        self.history:   list[dict]        = []
        self.fit_time:  float             = 0.0
        self.final_mre: float             = float('nan')
        self._last_grad: np.ndarray | None = None

    def _phi_and_grad(self, lam: np.ndarray):
        """Dual objective Phi(lambda) and gradient nabla Phi = E_{p_lam}[f] - alpha."""
        log_unnorm = self.F @ lam
        log_Z      = logsumexp(log_unnorm)
        p          = np.exp(log_unnorm - log_Z)
        alpha_hat  = self.F.T @ p
        grad       = alpha_hat - self.alphas
        obj        = log_Z - np.dot(lam, self.alphas)
        self._last_grad = grad          # <-- per il callback, evita il ricalcolo
        return obj, grad

    def fit(self, tol=1e-9, max_iter=1000,
            verbose=True) -> 'ExactMaxEntSolver':
        """
        Fit via L-BFGS-B.

        Parameters
        ----------
        tol : float — gradient tolerance for L-BFGS convergence
        max_iter : int — maximum L-BFGS iterations
        verbose : bool

        Returns
        -------
        self : ExactMaxEntSolver (fitted)
        """
        self.history = []
        lam0         = np.zeros(self.cs.m)
        t_start      = time.time()
        step         = [0]
                
        def get_probs(self) -> np.ndarray:
            """Return p_{lambda} over the full tuple space X."""
            if self.lambdas is None:
                raise RuntimeError("Call fit() before get_probs().")
                log_unnorm = self.F @ self.lambdas
                return np.exp(log_unnorm - logsumexp(log_unnorm))

        def callback(lam):
            grad = self._last_grad      # gia' calcolato da L-BFGS: niente matvec
            mre = np.mean(np.abs(grad) / (self.alphas + 1e-12))
            self.history.append({'iter': step[0], 'mre': float(mre)})
            step[0] += 1
            if verbose and step[0] % 10 == 0:
                print(f"  [Exact] iter {step[0]:4d}  MRE={mre:.6f}")

        res = minimize(
            self._phi_and_grad, lam0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol},
            callback=callback
        )

        self.lambdas   = res.x
        self.fit_time  = time.time() - t_start
        _, grad        = self._phi_and_grad(self.lambdas)
        self.final_mre = float(np.mean(
            np.abs(grad) / (self.alphas + 1e-12)
        ))

        if verbose:
            print(f"  [Exact] Converged: MRE={self.final_mre:.2e}  "
                  f"t={self.fit_time:.2f}s")
        return self
                
    def get_probs(self) -> np.ndarray:
        """Return p_{lambda} over the full tuple space X."""
        if self.lambdas is None:
            raise RuntimeError("Call fit() before get_probs().")
        log_unnorm = self.F @ self.lambdas
        return np.exp(log_unnorm - logsumexp(log_unnorm))

    def __repr__(self):
        status = (f"fitted, MRE={self.final_mre:.2e}"
                  if self.lambdas is not None else "not fitted")
        return f"ExactMaxEntSolver(K={self.K}, m={self.cs.m}, {status})"


class RakingSolver:
    """
    Generalised raking (Iterative Proportional Fitting, IPF).

    Reweights a fixed sample of N individuals to match aggregate
    constraint targets. Each iteration rescales weights for individuals
    matching each constraint by alpha_j / alpha_hat_j.

    Properties
    ----------
    - Exact on training constraints by algebraic construction (MRE=0).
    - Fast: O(N * m) per cycle.
    - Structural limitation: effective sample size N_eff = 1/sum(w^2)
      collapses exponentially with K, from N_eff/N ~ 7% at K=12
      to N_eff/N ~ 0.12% at K>=40.
    - Fails to satisfy training constraints for K >= ~40 with ternary
      constraints (MRE > 0).

    Parameters
    ----------
    cs : ConstraintSet
    """

    def __init__(self, cs: ConstraintSet):
        self.cs     = cs
        self.alphas = cs.alphas_array

        self.weights:   np.ndarray | None = None
        self.history:   list[dict]        = []
        self.fit_time:  float             = 0.0
        self.final_mre: float             = float('nan')
        self.n_iters:   int               = 0
        

    def fit(self, population: np.ndarray,
            max_iter: int = 1000,
            tol: float    = 1e-6,
            window: int   = 30,
            verbose: bool = False) -> 'RakingSolver':
        """
        Fit raking weights on a fixed population sample.

        Parameters
        ----------
        population : (N, K) int array — initial synthetic sample
        max_iter : int — maximum IPF cycles
        tol : float — convergence tolerance on max weight change
        window : int — window for convergence check
        verbose : bool

        Returns
        -------
        self : RakingSolver (fitted)
        """
        N          = len(population)
        weights    = np.ones(N, dtype=np.float64) / N
        t_start    = time.time()
        self.history = []

        for it in range(max_iter):
            w_prev = weights.copy()

            for j in range(self.cs.m):
                attrs = self.cs.attrs_list[j]
                vals  = self.cs.vals_list[j]
                mask  = np.all(
                    population[:, attrs] == vals[np.newaxis, :], axis=1
                )
                alpha_hat = float(weights[mask].sum())
                if alpha_hat > 1e-15:
                    weights[mask] *= self.alphas[j] / alpha_hat

            # Renormalise
            weights /= weights.sum()

            # MRE on training constraints
            alpha_hat_all = np.array([
                float(weights[np.all(
                    population[:, self.cs.attrs_list[j]] ==
                    self.cs.vals_list[j][np.newaxis, :], axis=1
                )].sum())
                for j in range(self.cs.m)
            ])
            mre = float(np.mean(
                np.abs(alpha_hat_all - self.alphas) / (self.alphas + 1e-12)
            ))

            self.history.append({'iter': it, 'mre': mre})

            if verbose and it % 10 == 0:
                print(f"  [Raking] iter {it:4d}  MRE={mre:.6f}")

            if np.max(np.abs(weights - w_prev)) < tol:
                break

        self.weights   = weights
        self.fit_time  = time.time() - t_start
        self.final_mre = self.history[-1]['mre']
        self.n_iters   = len(self.history)
        return self

    def effective_n(self) -> float:
        """Effective sample size N_eff = 1 / sum(w^2)."""
        if self.weights is None:
            raise RuntimeError("Call fit() first.")
        w = self.weights / self.weights.sum()
        return float(1.0 / np.sum(w ** 2))

    def gini(self) -> float:
        """Gini coefficient of the weight distribution."""
        if self.weights is None:
            raise RuntimeError("Call fit() first.")
        w    = np.sort(self.weights / self.weights.sum())
        n    = len(w)
        idx  = np.arange(1, n + 1)
        return float((2 * np.sum(idx * w) - (n + 1)) / n)

    def __repr__(self):
        status = (f"fitted, MRE={self.final_mre:.2e}, "
                  f"N_eff={self.effective_n():.0f}"
                  if self.weights is not None else "not fitted")
        return f"RakingSolver(K={self.cs.K}, {status})"
