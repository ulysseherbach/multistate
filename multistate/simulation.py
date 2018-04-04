"""
Simulation
----------

Compute sample paths for multi-state promoters
using the continuous (PDMP) framework
"""

import numpy as np
from multistate.promoters import transition_matrix

def pdmp_flow(time, state, d0):
    """Deterministic part of the PDMP model."""
    i, x = state[0]-1, state[1].copy()
    n, sx = np.size(x), 0
    ### Explicit solution of the ODE generating the flow
    for j in range(n):
        if (j != i):
            x[j] *= np.exp(-time*d0)
            sx += x[j]
    x[i] = 1 - sx
    return x

def promoter_step(state, k):
    """Compute the next jump and the next step."""
    i, x = state[0]-1, state[1]
    n = np.size(x)
    tau = -k[i,i] # Leaving rate from state i
    ### 1. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    ### 2. Update the promoter state
    v = np.zeros(n) # Probabilities for possible transitions
    v[:] = k[:,i]/tau
    v[i] = 0
    E = np.random.choice(n, p=v) + 1
    return E, U

def simulate(rate, timepoints, init_state=None, d0=1):
    """Exact simulation of the multistate PDMP promoter model."""
    if (np.size(timepoints) == 1):
        timepoints = np.array([timepoints])
    if np.any(timepoints != np.sort(timepoints)):
        print('Error: timepoints must be in increasing order')
        return None
    H = transition_matrix(rate)
    n = np.size(H[0])
    types = [('E','uint8'), ('X','float64',n)]
    ### Initialization
    T, sim = 0, []
    E, X = 1, np.zeros(n)
    if init_state is None: X[n-1] = 1
    elif (np.sum(init_state[1]) == 1):
        E = init_state[0]
        X[:] = init_state[1]
    state = (E,X)
    ### The core loop for simulation and recording
    Told, state_old = T, state
    for t in timepoints:
        while (t >= T):
            Told, state_old = T, state
            E, U = promoter_step(state, H)
            X = pdmp_flow(U, state, d0)
            state = (E,X)
            T += U
        sim += [(state_old[0],pdmp_flow(t-Told, state_old, d0))]
    return np.array(sim, dtype=types)