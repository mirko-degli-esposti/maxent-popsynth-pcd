"""
exact_marginals.py
------------------
Analytical computation of all marginal targets for Syn-ISTAT.

compute_exact_marginals()
    Returns ALPHA_EXACT: dict mapping attribute-name tuples to
    numpy arrays of exact marginal probabilities, computed by
    factor marginalisation over the CPTs without Monte Carlo sampling.

build_cs_from_alpha()
    Converts ALPHA_EXACT into a ConstraintSet for solver input.

All 31 marginals (15 unary + 13 binary + 3 ternary) normalise to 1
within machine precision (< 1e-14).
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.constraint_set import ConstraintSet
from src.syn_istat.attr_meta import (
    ATTR_META, DOMAIN_SIZES_SYNTH,
    marginals,
    age_marital, employment_commute, age_diet, age_alcohol,
    age_activity, household_children,
    triple_sae, triple_eei, triple_rct,
)


# ------------------------------------------------------------------ #
#  CPT conversion helpers                                               #
# ------------------------------------------------------------------ #

def _marg_to_vec(name):
    """Convert marginals[name] (dict) to normalised float64 array."""
    vals  = ATTR_META[name]['vals']
    probs = np.array([marginals[name][v] for v in vals], dtype=np.float64)
    return probs / probs.sum()


def _cond1_to_mat(out_name, cond_name, table):
    """
    Convert P(out | cond) from nested dict to array of shape
    (d_cond, d_out), with each row normalised.
    """
    d_c = len(ATTR_META[cond_name]['vals'])
    d_o = len(ATTR_META[out_name]['vals'])
    arr = np.zeros((d_c, d_o), dtype=np.float64)
    for i, vc in enumerate(ATTR_META[cond_name]['vals']):
        if vc not in table:
            continue
        for j, vo in enumerate(ATTR_META[out_name]['vals']):
            arr[i, j] = table[vc].get(vo, 0.0)
        s = arr[i].sum()
        if s > 0:
            arr[i] /= s
    return arr


def _cond2_to_arr(out_name, cond1_name, cond2_name, table):
    """
    Convert P(out | cond1, cond2) from nested dict to array of shape
    (d_c1, d_c2, d_out), with each slice [i,j,:] normalised.
    """
    d_c1 = len(ATTR_META[cond1_name]['vals'])
    d_c2 = len(ATTR_META[cond2_name]['vals'])
    d_o  = len(ATTR_META[out_name]['vals'])
    arr  = np.zeros((d_c1, d_c2, d_o), dtype=np.float64)
    for i, v1 in enumerate(ATTR_META[cond1_name]['vals']):
        if v1 not in table:
            continue
        for j, v2 in enumerate(ATTR_META[cond2_name]['vals']):
            if v2 not in table[v1]:
                continue
            for k, vo in enumerate(ATTR_META[out_name]['vals']):
                arr[i, j, k] = table[v1][v2].get(vo, 0.0)
            s = arr[i, j].sum()
            if s > 0:
                arr[i, j] /= s
    return arr


# ------------------------------------------------------------------ #
#  Main function                                                        #
# ------------------------------------------------------------------ #

def compute_exact_marginals(verbose: bool = True) -> dict:
    """
    Compute all marginal targets alpha* for Syn-ISTAT analytically
    by factor marginalisation over the CPTs.

    Returns
    -------
    ALPHA_EXACT : dict mapping tuple(attribute_names) -> np.ndarray
        Each value is a normalised probability array:
        shape (d,)      for unary
        shape (d1, d2)  for binary
        shape (d1,d2,d3) for ternary
        Axes follow the order of the key tuple.
    """
    # ── Anchor marginals ──────────────────────────────────────────
    p_sex  = _marg_to_vec('sex')             # (2,)
    p_age  = _marg_to_vec('age')             # (4,)
    p_edu  = _marg_to_vec('education')       # (4,)
    p_hh   = _marg_to_vec('household_size')  # (4,)
    p_area = _marg_to_vec('residence_area')  # (3,)
    p_car  = _marg_to_vec('car_access')      # (3,)

    # ── Binary CPTs as numpy arrays ───────────────────────────────
    cpt_mar = _cond1_to_mat('marital',          'age',            age_marital)
    cpt_com = _cond1_to_mat('commute_time',      'employment',     employment_commute)
    cpt_die = _cond1_to_mat('diet_type',         'age',            age_diet)
    cpt_alc = _cond1_to_mat('alcohol_use',       'age',            age_alcohol)
    cpt_act = _cond1_to_mat('physical_activity', 'age',            age_activity)
    cpt_chi = _cond1_to_mat('has_children',      'household_size', household_children)

    # ── Ternary CPTs: T3 P(emp|sex,age), T1 P(inc|edu,emp), T2 P(tra|area,car)
    cpt_emp = _cond2_to_arr('employment',    'sex',           'age',          triple_sae)
    cpt_inc = _cond2_to_arr('income',        'education',     'employment',   triple_eei)
    cpt_tra = _cond2_to_arr('main_transport','residence_area','car_access',   triple_rct)

    # ── Derived marginal P(employment) ────────────────────────────
    p_emp = np.einsum('s,a,sae->e', p_sex, p_age, cpt_emp)   # (3,)

    A = {}

    # ── Unary (15) ────────────────────────────────────────────────
    A[('sex',)]               = p_sex.copy()
    A[('age',)]               = p_age.copy()
    A[('education',)]         = p_edu.copy()
    A[('household_size',)]    = p_hh.copy()
    A[('residence_area',)]    = p_area.copy()
    A[('car_access',)]        = p_car.copy()
    A[('marital',)]           = np.einsum('a,am->m', p_age, cpt_mar)
    A[('employment',)]        = p_emp.copy()
    A[('income',)]            = np.einsum(
        'd,s,a,sae,dei->i', p_edu, p_sex, p_age, cpt_emp, cpt_inc)
    A[('has_children',)]      = np.einsum('h,hc->c', p_hh, cpt_chi)
    A[('main_transport',)]    = np.einsum('a,c,act->t', p_area, p_car, cpt_tra)
    A[('commute_time',)]      = np.einsum('e,ec->c', p_emp, cpt_com)
    A[('diet_type',)]         = np.einsum('a,ad->d', p_age, cpt_die)
    A[('alcohol_use',)]       = np.einsum('a,al->l', p_age, cpt_alc)
    A[('physical_activity',)] = np.einsum('a,av->v', p_age, cpt_act)

    # ── Binary (13) ───────────────────────────────────────────────
    A[('age','marital')]             = np.einsum('a,am->am', p_age, cpt_mar)
    A[('age','employment')]          = np.einsum('a,s,sae->ae', p_age, p_sex, cpt_emp)
    A[('education','employment')]    = np.einsum(
        'd,s,a,sae->de', p_edu, p_sex, p_age, cpt_emp)
    A[('employment','income')]       = np.einsum(
        's,a,sae,d,dei->ei', p_sex, p_age, cpt_emp, p_edu, cpt_inc)
    A[('household_size','has_children')] = np.einsum('h,hc->hc', p_hh, cpt_chi)
    A[('residence_area','main_transport')] = np.einsum(
        'a,c,act->at', p_area, p_car, cpt_tra)
    A[('car_access','main_transport')]   = np.einsum(
        'a,c,act->ct', p_area, p_car, cpt_tra)
    A[('age','alcohol_use')]         = np.einsum('a,al->al', p_age, cpt_alc)
    A[('sex','employment')]          = np.einsum('s,a,sae->se', p_sex, p_age, cpt_emp)
    A[('sex','income')]              = np.einsum(
        's,a,d,sae,dei->si', p_sex, p_age, p_edu, cpt_emp, cpt_inc)
    A[('employment','commute_time')] = np.einsum('e,ec->ec', p_emp, cpt_com)
    A[('age','diet_type')]           = np.einsum('a,ad->ad', p_age, cpt_die)
    A[('age','physical_activity')]   = np.einsum('a,av->av', p_age, cpt_act)

    # ── Ternary (3) ───────────────────────────────────────────────
    A[('education','employment','income')] = np.einsum(
        'd,s,a,sae,dei->dei', p_edu, p_sex, p_age, cpt_emp, cpt_inc)
    A[('residence_area','car_access','main_transport')] = np.einsum(
        'a,c,act->act', p_area, p_car, cpt_tra)
    A[('sex','age','employment')] = np.einsum('s,a,sae->sae', p_sex, p_age, cpt_emp)

    # ── Sanity checks ─────────────────────────────────────────────
    if verbose:
        print("=== compute_exact_marginals() — sanity checks ===")
        errs = [key for key, arr in A.items() if abs(arr.sum() - 1.0) > 1e-10]
        if errs:
            for k in errs:
                print(f"  {k}: sum={A[k].sum():.8f}  <- ERROR")
        else:
            print(f"  All {len(A)} marginals normalised to 1 within 1e-10 ✓")

        p_sex_check = A[('sex','age','employment')].sum(axis=(1, 2))
        print(f"  P(sex) from T3 vs anchor: max_err="
              f"{np.abs(p_sex_check - p_sex).max():.2e}")

        p_emp_check = A[('age','employment')].sum(axis=0)
        print(f"  P(emp) from B2 vs direct: max_err="
              f"{np.abs(p_emp_check - p_emp).max():.2e}")

        n_u = sum(1 for k in A if len(k) == 1)
        n_b = sum(1 for k in A if len(k) == 2)
        n_t = sum(1 for k in A if len(k) == 3)
        print(f"  Total: {len(A)} constraints "
              f"({n_u} unary, {n_b} binary, {n_t} ternary)")

    return A


# ------------------------------------------------------------------ #
#  ConstraintSet builder                                                #
# ------------------------------------------------------------------ #

def build_cs_from_alpha(alpha_dict: dict,
                         arities: set = None) -> ConstraintSet:
    """
    Build a ConstraintSet from ALPHA_EXACT.

    Parameters
    ----------
    alpha_dict : output of compute_exact_marginals()
    arities : None -> all constraints
              {1}  -> unary only
              {1,2}-> unary + binary
              {3}  -> ternary only

    Returns
    -------
    cs : ConstraintSet with m atomic indicator constraints
    """
    cs = ConstraintSet(DOMAIN_SIZES_SYNTH)
    for key, arr in alpha_dict.items():
        if arities is not None and len(key) not in arities:
            continue
        attr_indices = [ATTR_META[name]['idx'] for name in key]
        for cell_idx in np.ndindex(*arr.shape):
            prob = float(arr[cell_idx])
            if prob < 1e-12:
                continue
            cs.add(
                attrs=attr_indices,
                vals=list(cell_idx),
                alpha=prob,
            )
    return cs


def build_syn_istat_constraint_sets(verbose: bool = True):
    """
    Build the three standard Syn-ISTAT constraint sets.

    Returns
    -------
    cs_full    : ConstraintSet with all 31 constraints (training A-ISTAT-1,3)
    cs_train28 : ConstraintSet with 28 unary+binary constraints (training A-ISTAT-2)
    cs_held3   : ConstraintSet with 3 ternary constraints (held-out A-ISTAT-2)
    """
    ALPHA_EXACT = compute_exact_marginals(verbose=verbose)
    cs_full     = build_cs_from_alpha(ALPHA_EXACT)
    cs_train28  = build_cs_from_alpha(ALPHA_EXACT, arities={1, 2})
    cs_held3    = build_cs_from_alpha(ALPHA_EXACT, arities={3})
    if verbose:
        print(f"\ncs_full    : {cs_full.summary()}")
        print(f"\ncs_train28 : {cs_train28.summary()}")
        print(f"\ncs_held3   : {cs_held3.summary()}")
        print(f"\n|X| = {np.prod(DOMAIN_SIZES_SYNTH):,}  (non-enumerable)")
    return cs_full, cs_train28, cs_held3
