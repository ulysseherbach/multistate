"""
Refractory
----------

Compute exact stationary distributions for refractory promoters
"""
import numpy as np
from mpmath import fac, power, exp, loggamma, hyper, meijerg
from scipy.linalg import expm
from multistate.promoters import transition_matrix

### "High-level" functions
def eigenvalues(rate, onstate=1):
    """Return the relevant eigenvalues for the density.
    - rate[i,j] is the rate of transition i -> j
    - state = 1,...,G is the reference (active) state"""
    K = transition_matrix(rate)
    n = np.size(K[:,0])
    ### Check the active state
    if onstate not in set(range(1,n+1)):
        print('Error: the active state must be in transitions')
        return None
    i = onstate - 1
    l = [j for j in range(i)] + [j for j in range(i+1,n)]
    A = K[l][:,l] # Remove i-th lign and i-th column
    u = np.sort(np.linalg.eigvals(-A))
    v = np.sort(np.linalg.eigvals(-K))[1:]
    return (u,v)

def dist_poisson(val, rate, onstate=1, scale=100):
    """RNA distribution in the discrete case."""
    a, b = eigenvalues(rate, onstate)
    plist, s = [], scale
    N = np.size(a) # Number of inactive states
    if (np.size(val) == 1):
        val = np.array([val], dtype=float)
    else:
        val = np.array(val, dtype=float)
    for m in val:
        am = [x + m for x in a]
        bm = [x + m for x in b]
        c = 0
        for k in range(N):
            c += loggamma(am[k]) - loggamma(a[k])
            c += loggamma(b[k]) - loggamma(bm[k])
        ### Precise computation step
        p = exp(c) * power(s, m) * hyper(am, bm, -s) / fac(m)
        plist.append(p.real)
    if (len(plist) == 1):
        return plist[0]
    else:
        return np.array(plist)

def dist_pdmp(val, rate, onstate=1, scale=1):
    """RNA distribution in the continuous case (PDMP)."""
    a, b = eigenvalues(rate, onstate)
    plist, s = [], scale
    N = np.size(a) # Number of inactive states
    if (np.size(val) == 1):
        val = np.array([val], dtype=float)
    else:
        val = np.array(val, dtype=float)
    c = 0
    for k in range(N):
        c += loggamma(b[k]) - loggamma(a[k])
    for x in val:
        ### Precise computation step
        p = exp(c) * meijerg([[],list(b-1)], [list(a-1),[]], x/s) / s
        plist.append(p.real)
    if (len(plist) == 1):
        return plist[0]
    else:
        return np.array(plist)

def dist_inactive(val, rate, onstate=1):
    """Distribution of the inactive period."""
    K = transition_matrix(rate)
    n = np.size(K[0])
    if onstate not in set(range(1,n+1)):
        return None
    if (np.size(val) == 1):
        val = np.array([val])
    i = onstate - 1
    p = K[:,i]
    p[i] = 0
    p = p/np.sum(p)
    K[:,i] = 0
    plist = []
    for t in val:
        v = np.dot(K, np.dot(expm(t*K), p))
        plist.append(v[i])
    if (len(plist) == 1):
        return plist[0]
    else:
        return np.array(plist)

def dist_active(val, rate, onstate=1):
    """Distribution of the active period."""
    K = transition_matrix(rate)
    n = np.size(K[0])
    if onstate not in set(range(1,n+1)):
        return None
    if (np.size(val) == 1):
        val = np.array([val])
    i = onstate - 1
    tau = -K[i,i]
    plist = [tau*np.exp(-t*tau) for t in val]
    if (len(plist) == 1):
        return plist[0]
    else:
        return np.array(plist)
