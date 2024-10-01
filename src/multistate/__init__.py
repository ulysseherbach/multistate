"""
Stochastic gene expression with multistate promoters
====================================================

This package aims at efficiently computing the analytical stationary
distribution for a large class of refractory promoters, as well as
performing exact stochastic simulations for general promoters in both
standard (SSA) and hybrid (PDMP) formalisms.

See https://github.com/ulysseherbach/multistate for documentation.
"""
from importlib.metadata import version as _version
from multistate.promoters import twostate, cyclic, dirichlet
from multistate.simulation import sim_pdmp, sim_ssa, conditional_pdmp

__all__ = ['twostate', 'cyclic', 'dirichlet']
__all__ += ['sim_pdmp', 'sim_ssa', 'conditional_pdmp']

__author__ = 'Ulysse Herbach'

try:
    __version__ = _version('multistate')
except Exception:
    __version__ = 'unknown version'
