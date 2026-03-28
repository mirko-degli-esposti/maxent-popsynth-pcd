"""
generators.py
-------------
Synthetic benchmark generators for controlled MaxEnt validation.

Classes
-------
WuGenerator
    Controlled benchmark following the experimental protocol of
    Wu et al. (2018). Generates synthetic populations with planted
    patterns of known frequency.

PlantedExpFamilyGenerator
    Ground-truth generator with analytically known lambda*.
    Enables exact KL and parameter-distance evaluation for K <= ~14.

Reference:
    Wu et al. (2018). Generating Realistic Synthetic Population Datasets.
    ACM Transactions on Knowledge Discovery from Data, 12(4).
"""

import numpy as np
from itertools import product as cart_product
from scipy.special import logsumexp
from .constraint_set import ConstraintSet


class WuGenerator:
    """
    Synthetic benchmark generator following Wu et al. (2018).

    Constructs a controlled ground truth as follows:
    (i)  Each attribute k is assigned a random marginal over D_k.
    (ii) A set of P binary or ternary planted patterns (S_p, v_p, nu_p)
         is drawn, each with target frequency nu_p in (0.05, 0.35).
    (iii) Each synthetic individual independently activates each
         pattern with probability nu_p (Bernoulli draw), receiving the
         prescribed values for x_{S_p}; attributes not covered by any
         activated pattern are sampled from their marginals.

    The constraint set is then the empirical frequencies computed from
    N_data such individuals. This yields, by construction, a consistent
    set of constraints with known planted structure.

    Parameters
    ----------
    K : int
        Number of attributes.
    domain_range : tuple(int, int)
        Range (min, max) for the number of values per attribute.
    n_patterns : int
        Number of patterns to plant.
    pattern_arity : int
        Arity of planted patterns (2 = pairs, 3 = triples).
    seed : int
    """

    def __init__(self, K: int, domain_range=(2, 4), n_patterns=10,
                 pattern_arity=2, seed=42):
        self.rng          = np.random.default_rng(seed)
        self.K            = K
        self.pattern_arity = pattern_arity

        self.domain_sizes = self.rng.integers(
            domain_range[0], domain_range[1] + 1, size=K
        ).astype(np.int32)

        # Independent marginals for each attribute
        self.marginals = []
        for k in range(K):
            raw = self.rng.random(self.domain_sizes[k]) + 0.1
            self.marginals.append(raw / raw.sum())

        self.patterns = self._build_patterns(n_patterns, pattern_arity)

    def _build_patterns(self, n_patterns: int, arity: int) -> list:
        """Build patterns on disjoint attributes with random frequencies."""
        patterns  = []
        attr_pool = self.rng.permutation(self.K).tolist()
        i = 0
        while len(patterns) < n_patterns and i + arity <= len(attr_pool):
            attrs = tuple(sorted(attr_pool[i: i + arity]))
            vals  = tuple(
                int(self.rng.integers(0, self.domain_sizes[a])) for a in attrs
            )
            freq = float(self.rng.uniform(0.05, 0.35))
            patterns.append({'attrs': attrs, 'vals': vals, 'freq': freq})
            i += arity
        return patterns

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate n_samples categorical tuples.

        Returns
        -------
        data : (n_samples, K) int32 array
        """
        data = np.zeros((n_samples, self.K), dtype=np.int32)
        for k in range(self.K):
            data[:, k] = self.rng.choice(
                self.domain_sizes[k], size=n_samples, p=self.marginals[k]
            )
        for pat in self.patterns:
            attrs    = np.array(pat['attrs'], dtype=np.int32)
            vals     = np.array(pat['vals'],  dtype=np.int32)
            activate = self.rng.random(n_samples) < pat['freq']
            data[np.ix_(activate, attrs)] = vals[np.newaxis, :]
        return data

    def extract_constraints(self, data: np.ndarray,
                             include_unary=True,
                             include_planted_binary=True,
                             include_all_binary=False,
                             include_ternary=False) -> ConstraintSet:
        """
        Extract constraints from the generated dataset.

        By default includes: unary marginals + planted patterns.
        """
        cs = ConstraintSet(self.domain_sizes)
        pattern_attrs = []

        if include_unary:
            pattern_attrs += [(k,) for k in range(self.K)]
        if include_planted_binary:
            pattern_attrs += [p['attrs'] for p in self.patterns]
        if include_all_binary and not include_planted_binary:
            pattern_attrs += [
                (i, j)
                for i in range(self.K)
                for j in range(i + 1, self.K)
            ]
        if include_ternary:
            pattern_attrs += [
                (i, j, k)
                for i in range(self.K)
                for j in range(i + 1, self.K)
                for k in range(j + 1, self.K)
            ]

        seen, unique = set(), []
        for p in pattern_attrs:
            key = tuple(sorted(p))
            if key not in seen:
                seen.add(key)
                unique.append(key)

        cs.add_from_data(data, unique)
        return cs

    def describe(self) -> str:
        lines = [
            f"WuGenerator: K={self.K}",
            f"  domain_sizes: {self.domain_sizes.tolist()}",
            f"  |X| = {np.prod(self.domain_sizes):,}",
            f"  Planted patterns ({len(self.patterns)}):",
        ]
        for i, p in enumerate(self.patterns):
            lines.append(
                f"    [{i}] attrs={p['attrs']} vals={p['vals']} "
                f"freq={p['freq']:.3f}"
            )
        return "\n".join(lines)


class PlantedExpFamilyGenerator:
    """
    Analytically exact ground-truth generator for MaxEnt validation.

    Procedure:
    1. Selects random K attributes with small domains.
    2. Draws m random constraint patterns.
    3. Assigns lambda_j* ~ Uniform[-scale, +scale].
    4. Computes p_{lambda*} by exact enumeration.
    5. Computes alpha_j* = E_{p_{lambda*}}[f_j] (exact targets).

    Enables exact computation of:
    - KL(p_{lambda*} || p_{lambda_MCMC})
    - ||lambda_MCMC - lambda*|| / ||lambda*||

    Works for K <= ~14 (|X| <= ~10^6).

    Parameters
    ----------
    domain_sizes : array-like of int
    seed : int
    """

    def __init__(self, domain_sizes, seed=42):
        self.rng          = np.random.default_rng(seed)
        self.domain_sizes = np.array(domain_sizes, dtype=np.int32)
        self.K            = len(domain_sizes)
        X_size            = int(np.prod(domain_sizes))

        if X_size > 5_000_000:
            raise ValueError(
                f"|X|={X_size:,} too large for exact enumeration. "
                f"Use smaller K or domain sizes."
            )

        ranges = [range(d) for d in self.domain_sizes]
        self.all_tuples = np.array(list(cart_product(*ranges)), dtype=np.int32)
        self.X_size = len(self.all_tuples)

        self.cs:            ConstraintSet | None = None
        self.lambdas_true:  np.ndarray    | None = None
        self.probs:         np.ndarray    | None = None
        self.log_probs:     np.ndarray    | None = None
        self.F:             np.ndarray    | None = None
        self.log_Z:         float                = 0.0

    def plant_constraints(self, n_constraints: int, arity=2,
                          lambda_scale=1.5) -> tuple[ConstraintSet, np.ndarray]:
        """
        Plant n_constraints constraints with random lambda*.

        Returns
        -------
        cs : ConstraintSet with alpha_j = E_{p_{lambda*}}[f_j]
        lambdas_true : (m,) array of true lambda*
        """
        cs = ConstraintSet(self.domain_sizes)
        for _ in range(n_constraints):
            attrs = tuple(sorted(
                self.rng.choice(self.K, size=arity, replace=False).tolist()
            ))
            vals = tuple(
                int(self.rng.integers(0, self.domain_sizes[a])) for a in attrs
            )
            cs.add(list(attrs), list(vals), alpha=0.0)

        lambdas_true = self.rng.uniform(-lambda_scale, lambda_scale, size=cs.m)

        F           = cs.build_indicator_matrix(self.all_tuples)
        log_unnorm  = F @ lambdas_true
        log_Z       = logsumexp(log_unnorm)
        log_probs   = log_unnorm - log_Z
        probs       = np.exp(log_probs)

        true_alphas = probs @ F
        for j in range(cs.m):
            cs.alphas[j] = float(true_alphas[j])

        self.cs           = cs
        self.lambdas_true = lambdas_true
        self.F            = F
        self.log_Z        = log_Z
        self.log_probs    = log_probs
        self.probs        = probs
        return cs, lambdas_true

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from p_{lambda*} (requires plant_constraints called first)."""
        if self.probs is None:
            raise RuntimeError("Call plant_constraints() before sample().")
        idx = self.rng.choice(self.X_size, size=n_samples, p=self.probs)
        return self.all_tuples[idx]

    def entropy(self) -> float:
        """Shannon entropy of p_{lambda*} in nats."""
        if self.probs is None:
            raise RuntimeError("Call plant_constraints() before entropy().")
        return float(-np.sum(self.probs * self.log_probs))

    def kl_from(self, lambdas_approx: np.ndarray) -> float:
        """
        KL(p_{lambda*} || p_{lambda_approx}).

        Parameters
        ----------
        lambdas_approx : (m,) array — approximate lambda from solver
        """
        if self.F is None:
            raise RuntimeError("Call plant_constraints() first.")
        log_unnorm_approx = self.F @ lambdas_approx
        log_Z_approx = logsumexp(log_unnorm_approx)
        log_p_approx = log_unnorm_approx - log_Z_approx
        return float(np.sum(self.probs * (self.log_probs - log_p_approx)))

    def describe(self) -> str:
        lines = [
            f"PlantedExpFamilyGenerator: K={self.K}",
            f"  domain_sizes: {self.domain_sizes.tolist()}",
            f"  |X| = {self.X_size:,}",
        ]
        if self.cs is not None:
            lines += [
                f"  Planted constraints: {self.cs.m}",
                f"  lambda* in [{self.lambdas_true.min():.3f}, "
                f"{self.lambdas_true.max():.3f}]",
                f"  H(p_lambda*) = {self.entropy():.4f} nats",
            ]
        return "\n".join(lines)
