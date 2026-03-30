"""
syn_istat
---------
Syn-ISTAT benchmark: K=15 Italian demographic benchmark
with analytically exact marginal targets.
"""

from .attr_meta import (
    ATTR_META, ATTR_NAMES_SYNTH, DOMAIN_SIZES_SYNTH, K_SYNTH,
    marginals,
)
from .exact_marginals import (
    compute_exact_marginals,
    build_cs_from_alpha,
    build_syn_istat_constraint_sets,
)

__all__ = [
    'ATTR_META', 'ATTR_NAMES_SYNTH', 'DOMAIN_SIZES_SYNTH', 'K_SYNTH',
    'marginals',
    'compute_exact_marginals',
    'build_cs_from_alpha',
    'build_syn_istat_constraint_sets',
]
