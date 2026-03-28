"""
constraint_set.py
-----------------
Core data structure for MaxEnt population synthesis constraints.

Each constraint j is a pattern indicator:
    f_j(x) = 1[ x_{S_j} = v_j ]
with empirical target alpha_j = E_p[f_j].

Reference:
    Wu et al. (2018). Generating Realistic Synthetic Population Datasets.
    ACM Transactions on Knowledge Discovery from Data, 12(4).
    Pachet & Zucker (2026). Maximum Entropy Relaxation of Multi-Way
    Cardinality Constraints for Synthetic Population Generation.
"""

import numpy as np


class ConstraintSet:
    """
    Collection of MaxEnt constraints for a categorical problem.

    Parameters
    ----------
    domain_sizes : array-like of int
        Number of values for each attribute (length K).

    Attributes
    ----------
    K : int
        Number of attributes.
    m : int
        Total number of atomic constraints (read-only property).
    attrs_list : list of np.ndarray
        Attribute indices for each constraint (sorted).
    vals_list : list of np.ndarray
        Required values for each constraint.
    alphas : list of float
        Target frequencies E_p[f_j].
    """

    def __init__(self, domain_sizes):
        self.domain_sizes = np.array(domain_sizes, dtype=np.int32)
        self.K = len(domain_sizes)
        self.attrs_list: list[np.ndarray] = []
        self.vals_list:  list[np.ndarray] = []
        self.alphas:     list[float]      = []

    # ------------------------------------------------------------------ #
    #  Adding constraints                                                   #
    # ------------------------------------------------------------------ #

    def add(self, attrs, vals, alpha: float):
        """Add a single constraint (pattern, value, target)."""
        attrs = np.array(attrs, dtype=np.int32)
        vals  = np.array(vals,  dtype=np.int32)
        order = np.argsort(attrs)
        self.attrs_list.append(attrs[order])
        self.vals_list.append(vals[order])
        self.alphas.append(float(alpha))

    # kept for backward compatibility
    add_constraint = add

    def add_from_data(self, data: np.ndarray, pattern_attrs_list):
        """
        Extract empirical frequencies from `data` for the given patterns.

        Parameters
        ----------
        data : (n_samples, K) int array
        pattern_attrs_list : iterable of attribute index tuples
        """
        n = len(data)
        for attrs in pattern_attrs_list:
            attrs_sorted = tuple(sorted(attrs))
            sub = data[:, list(attrs_sorted)]
            combos, counts = np.unique(sub, axis=0, return_counts=True)
            for combo, count in zip(combos, counts):
                self.add(list(attrs_sorted), combo.tolist(), count / n)

    # ------------------------------------------------------------------ #
    #  Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def m(self) -> int:
        """Total number of atomic constraints."""
        return len(self.alphas)

    @property
    def alphas_array(self) -> np.ndarray:
        return np.array(self.alphas, dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  Matrices and lookup for solvers                                      #
    # ------------------------------------------------------------------ #

    def build_indicator_matrix(self, all_tuples: np.ndarray) -> np.ndarray:
        """
        Build F in {0,1}^{|X| x m}: F[x, j] = f_j(x).

        Used by the exact solver to compute E_p[f] in vectorised form.

        Parameters
        ----------
        all_tuples : (|X|, K) int array — full enumeration of X.

        Returns
        -------
        F : (|X|, m) float64 array
        """
        X_size = len(all_tuples)
        F = np.zeros((X_size, self.m), dtype=np.float64)
        for j in range(self.m):
            attrs = self.attrs_list[j]
            vals  = self.vals_list[j]
            match = np.all(all_tuples[:, attrs] == vals[np.newaxis, :], axis=1)
            F[:, j] = match
        return F

    def build_attr_lookup(self) -> dict:
        """
        For each attribute k, build the list of entries:
            (j, v_k_required, other_attrs, other_vals)
        where j is the constraint index, v_k is the required value for k,
        and other_* are the remaining attributes/values in the pattern.

        Used by the Gibbs sampler to compute conditional distributions
        efficiently without iterating over all constraints at each step.

        Returns
        -------
        lookup : dict mapping k -> list of (j, v_k, other_attrs, other_vals)
        """
        lookup = {k: [] for k in range(self.K)}
        for j in range(self.m):
            attrs = self.attrs_list[j]
            vals  = self.vals_list[j]
            for pos, k in enumerate(attrs):
                v_k = int(vals[pos])
                other_mask  = np.arange(len(attrs)) != pos
                other_attrs = attrs[other_mask]
                other_vals  = vals[other_mask]
                lookup[k].append((j, v_k, other_attrs, other_vals))
        return lookup

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        arities = [a.size for a in self.attrs_list]
        lines = [
            f"ConstraintSet: K={self.K}, m={self.m}",
            f"  Unary   (arity=1): {arities.count(1)}",
            f"  Binary  (arity=2): {arities.count(2)}",
            f"  Ternary (arity=3): {arities.count(3)}",
            f"  alpha in [{min(self.alphas):.4f}, {max(self.alphas):.4f}]",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (f"ConstraintSet(K={self.K}, m={self.m}, "
                f"domain_sizes={self.domain_sizes.tolist()})")
