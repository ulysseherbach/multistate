"""
Simulation
----------

Compute sample paths for multi-state promoters using:
- the classical (discrete) framework, aka SSA algorithm
- the PDMP (continuous) framework
"""

import numpy as np
from multistate.promoters import transition_matrix

def pdmp_flow(time, state, d0):
    """Deterministic part of the PDMP model."""
    i, x = state[0]-1, list(state[1])
    n, sx = np.size(x), 0
    ### Explicit solution of the ODE generating the flow
    for j in range(n):
        if (j != i):
            x[j] *= np.exp(-time*d0)
            sx += x[j]
    x[i] = 1 - sx
    return tuple(x)

def step_promoter(state, k):
    """Compute the next jump and the next step."""
    i = state[0] - 1
    n = np.size(state[1])
    tau = -k[i,i] # Leaving rate from state i
    ### 1. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    ### 2. Update the promoter state
    v = np.zeros(n) # Probabilities for possible transitions
    v[:] = k[:,i]/tau
    v[i] = 0
    E = np.random.choice(n, p=v) + 1
    return E, U

def sim_pdmp(rate, timepoints, init_state=None, d0=1):
    """Exact simulation of the PDMP multistate promoter model."""
    if (np.size(timepoints) == 1):
        timepoints = np.array([timepoints])
    if np.any(timepoints != np.sort(timepoints)):
        print('Error: timepoints must be in increasing order')
        return None
    H = transition_matrix(rate)
    n = np.size(H[0])
    types = [('E','int64'), ('X','float64',n)]
    ### Initialization
    T, sim = 0, []
    if init_state is None: 
        E, X = 1, (n-1)*(0,) + (1,)
    elif (np.sum(init_state[1]) == 1):
        E, X = init_state[0], tuple(init_state[1])
    else: print('Warning: init_state should sum to 1')
    state = (E,X)
    ### The core loop for simulation and recording
    for t in timepoints:
        while (t >= T):
            Told, state_old = T, state
            E, U = step_promoter(state, H)
            X = pdmp_flow(U, state, d0)
            state = (E,X)
            T += U
        sim += [(state_old[0],pdmp_flow(t-Told, state_old, d0))]
    return np.array(sim, dtype=types)

def step_ssa(state, k, u, d0):
    """Compute next jump for the basic multistate promoter model."""
    T, E, M = state
    n = np.size(k[0])
    ### Rates of possible reactions
    v, i = np.zeros(n+2), E-1
    v[:n], v[i] = k[:,i], 0 # Promoter transitions
    v[n] = u[i] # RNA creation
    v[n+1] = d0*M # RNA degradation
    ### 1. Draw the waiting time before the next jump
    tau = np.sum(v)
    U = np.random.exponential(scale=1/tau)
    T += U
    ### 2. Update the state
    r = np.random.choice(n+2, p=v/tau)
    if (r < n): E = r + 1
    elif (r == n): M += 1
    else: M -= 1
    return T, E, M, U

def sim_ssa(rate, u, time, init_state=(1,0), d0=1):
    """Exact simulation of the basic multistate promoter model."""
    H = transition_matrix(rate)
    types = [('T','float64'), ('E','int64'), ('M','int64')]
    ### Initialization
    T, (E, M) = 0, init_state
    sim = [(T, E, M)]
    ### The core loop for simulation and recording
    while (T < time):
        tstate = (T, E, M)
        T, E, M, U = step_ssa(tstate, H, u, d0)
        sim += [(T, E, M)]
    sim[-1] = (time, sim[-2][1], sim[-2][2])
    return np.array(sim, dtype=types)

def conditional_pdmp(timepoints, x0, jtraj, d0=1):
    """Compute the PDMP path conditionally on a promoter
    sample path (jtraj) and given an initial state (x0)."""
    if (np.size(timepoints) == 1):
        timepoints = np.array([timepoints])
    if np.any(timepoints != np.sort(timepoints)):
        print('Error: timepoints must be in increasing order')
        return None
    n = np.size(x0)
    types = [('E','int64'), ('X','float64',n)]
    ### Initialization
    T, sim = 0, []
    state = (jtraj['E'][0], tuple(x0))
    ### The core loop for simulation and recording
    k, l = -1, np.size(jtraj)
    for t in timepoints:
        while (t >= T):
            Told, state_old = T, state
            if (k < l-1): k += 1
            E, U = jtraj['E'][k], jtraj['U'][k]
            X = pdmp_flow(U, state, d0)
            state = (E,X)
            T += U
        sim += [(state_old[0],pdmp_flow(t-Told, state_old, d0))]
    return np.array(sim, dtype=types)

# def conditional_ssa(timepoints, m0, jtraj, u, d0=1):
#     """BETA: Compute a path of the classic model conditionally on a
#     promoter path (jtraj) and given an initial state (m0)."""
#     dt = 1e-1 * (1/d0)
#     n = np.max(jtraj['E'])
#     x0 = np.zeros(n)
#     # We need an initial condition such that u.x0 = 0
#     if (u[0] == 0): x0[0] = 1
#     else: x0[0], x0[1] = -u[1]/u[0], 1
#     x0 = x0/np.sum(x0)
#     types = [('E','int64'), ('M','float64')]
#     ### Initialization
#     T, sim = 0, []
#     state = (jtraj['E'][0], m0)
#     ### The core loop for simulation and recording
#     # for t in timepoints:
#     #     while (t >= T):
#     #         Told, (e0,m0) = T, state
#     #         traj = conditional_pdmp(dt, x0, jtraj, d0)
#     #         r1 = np.random.poisson(np.dot(traj['X'], u))
#     #         r2 = np.random.binomial(m0,np.exp(-d0*dt))
#     #         M = r1 + r2
#     #         state = (E,M)
#     #         T += dt
#     #     sim += [(state_old[0],pdmp_flow(t-Told, state_old, d0))]
#     # return np.array(sim, dtype=types)

### Utility functions
def get_jtraj(T, E):
    """Get minimal jumps and states from a given promoter sample path."""
    types = [('U','float64'), ('E','int64')]
    told, e = 0, E[0]
    jtraj = [(0, e)]
    for k, t in enumerate(T):
        if (E[k] != e):
            u, e = t - told, E[k]
            jtraj += [(u, e)]
            told = t
    return np.array(jtraj, dtype=types)

def simplify_prom(T, E):
    """Simplify a promoter sample path by removing phantom jumps."""
    types = [('T','float64'), ('E','int64')]
    e = E[0]
    traj = [(0, e)]
    for k, t in enumerate(T):
        if (E[k] != e) or (k == (np.size(T)-1)):
            e = E[k]
            traj += [(t, e)]
    return np.array(traj, dtype=types)
    