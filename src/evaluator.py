"""
evaluator.py
------------
Evaluation metrics and plotting utilities for GibbsPCDSolver validation.

Reference:
    Degli Esposti, M. (2026). Scalable Maximum Entropy Population Synthesis
    via Persistent Contrastive Divergence. arXiv:2503.XXXXX
"""

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt


class Evaluator:
    """
    Comparison metrics for GibbsPCDSolver validation experiments.

    All methods are static — no instantiation needed.
    """

    @staticmethod
    def mre(alpha_hat: np.ndarray, alpha_true: np.ndarray) -> float:
        """
        Mean Relative Error on constraints.

        MRE = (1/m) sum_j |alpha_hat_j - alpha_j| / alpha_j

        Parameters
        ----------
        alpha_hat : (m,) estimated constraint frequencies
        alpha_true : (m,) target constraint frequencies
        """
        return float(np.mean(
            np.abs(alpha_hat - alpha_true) / (alpha_true + 1e-12)
        ))

    @staticmethod
    def parameter_distance(lam_a: np.ndarray, lam_b: np.ndarray) -> float:
        """
        Normalised L2 distance between parameter vectors.

        ||lam_a - lam_b|| / ||lam_b||

        Note: this metric is unreliable when unary constraints are present
        (gauge non-identifiability). Use MRE and KL as primary diagnostics.
        """
        return float(
            np.linalg.norm(lam_a - lam_b) / (np.linalg.norm(lam_b) + 1e-12)
        )

    @staticmethod
    def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
        """
        KL(p || q) = sum_x p(x) log(p(x) / q(x)).

        Both p and q are renormalised internally.
        """
        p = np.clip(p, eps, None); p = p / p.sum()
        q = np.clip(q, eps, None); q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def effective_n(weights: np.ndarray) -> float:
        """
        Effective sample size N_eff = 1 / sum(w^2).

        For GibbsPCDSolver (uniform weights): N_eff = N.
        For raking: N_eff << N at large K.
        """
        w = weights / weights.sum()
        return float(1.0 / np.sum(w ** 2))

    @staticmethod
    def compute_gibbs_probs(gibbs, exact) -> np.ndarray:
        """
        Compute p_{lambda_MCMC} over the full tuple space X
        using the indicator matrix F from an ExactMaxEntSolver.

        Only feasible when |X| is enumerable (small-K experiments).

        Parameters
        ----------
        gibbs : GibbsPCDSolver (fitted)
        exact : ExactMaxEntSolver (fitted, provides F and all_tuples)

        Returns
        -------
        probs : (|X|,) probability array under lambda_MCMC
        """
        log_unnorm = exact.F @ gibbs.lambdas
        return np.exp(log_unnorm - logsumexp(log_unnorm))


# ------------------------------------------------------------------ #
#  Plotting utilities                                                   #
# ------------------------------------------------------------------ #

def plot_convergence(results: dict, exact_mre: float, title: str,
                     log_scale: bool = True,
                     figsize: tuple = (10, 4)):
    """
    MRE convergence curves for different pool sizes.

    Parameters
    ----------
    results : dict mapping N_pool (int) -> GibbsPCDSolver (fitted)
    exact_mre : float — reference MRE from ExactMaxEntSolver
    title : str — figure title prefix
    log_scale : bool — log y-axis
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for (N_pool, solver), color in zip(results.items(), colors):
        mres = [h['mre'] for h in solver.history]
        axes[0].plot(mres, color=color, lw=1.5, label=f'N={N_pool:,}')
    axes[0].axhline(exact_mre, color='k', ls='--', lw=1.5, label='Exact L-BFGS')
    axes[0].set_xlabel('Outer iteration')
    axes[0].set_ylabel('MRE')
    axes[0].set_title(f'{title} — MRE convergence')
    if log_scale:
        axes[0].set_yscale('log')
    axes[0].legend(fontsize=9)

    ns   = list(results.keys())
    mres = [results[n].final_mre for n in ns]
    axes[1].plot(ns, mres, 'o-', color='#2c7bb6', lw=1.5, ms=6)
    axes[1].axhline(exact_mre, color='k', ls='--', lw=1.5, label='Exact L-BFGS')
    axes[1].set_xlabel('N_pool')
    axes[1].set_ylabel('Final MRE')
    axes[1].set_title(f'{title} — MRE vs N_pool')
    axes[1].set_xscale('log')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_lambda_scatter(exact, gibbs,
                        title: str = 'Parameter recovery',
                        ax=None):
    """
    Scatter plot of lambda_exact vs lambda_MCMC with y=x reference line.

    Parameters
    ----------
    exact : ExactMaxEntSolver (fitted)
    gibbs : GibbsPCDSolver (fitted)
    title : str
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    lam_e = exact.lambdas
    lam_g = gibbs.lambdas
    ax.scatter(lam_e, lam_g, alpha=0.5, s=20, color='#3498db')
    lim = max(np.abs(lam_e).max(), np.abs(lam_g).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    ax.set_xlabel('lambda_exact')
    ax.set_ylabel('lambda_MCMC')
    ax.set_title(title)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    dist = Evaluator.parameter_distance(lam_g, lam_e)
    ax.text(0.05, 0.92, f'||Δλ||/||λ*|| = {dist:.3f}',
            transform=ax.transAxes, fontsize=9)
    return ax


def print_summary_table(label: str, exact,
                         gibbs_results: dict,
                         planted_probs: np.ndarray = None):
    """
    Print a summary table comparing GibbsPCDSolver runs against
    the exact L-BFGS reference.

    Parameters
    ----------
    label : str — experiment label
    exact : ExactMaxEntSolver (fitted)
    gibbs_results : dict mapping (N_pool, n_sweeps) -> GibbsPCDSolver
    planted_probs : (|X|,) array or None
        If provided, KL is computed against the planted ground truth
        p_{lambda*} rather than p_{lambda_exact}.
    """
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  Exact L-BFGS:  MRE={exact.final_mre:.5f}  t={exact.fit_time:.2f}s")
    print(f"{'─'*70}")
    print(f"  {'N_pool':>10} {'Sweeps':>8} {'MRE':>10} "
          f"{'||Δλ||/||λ||':>14} {'KL':>10} {'t(s)':>8}")
    print(f"{'─'*70}")

    p_exact = exact.get_probs()

    for (N_pool, n_sweeps), solver in gibbs_results.items():
        p_gibbs = Evaluator.compute_gibbs_probs(solver, exact)
        param_d = Evaluator.parameter_distance(solver.lambdas, exact.lambdas)
        if planted_probs is not None:
            kl_val  = Evaluator.kl(planted_probs, p_gibbs)
            ref_lbl = '†'
        else:
            kl_val  = Evaluator.kl(p_exact, p_gibbs)
            ref_lbl = ''
        print(f"  {N_pool:>10,} {n_sweeps:>8} {solver.final_mre:>10.5f} "
              f"{param_d:>14.5f} {kl_val:>10.6f}{ref_lbl} "
              f"{solver.fit_time:>8.1f}")

    if planted_probs is not None:
        print(f"  † KL against p_lambda* (analytical ground truth)")
    print(f"{'─'*70}")
