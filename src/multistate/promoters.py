"""
Some examples of promoter models
--------------------------------

NB: when modeling a refractory promoter (only one active state),
all functions make the convention that state 1 is the active one.
"""
import numpy as np

### General functions
def transition_matrix(rate):
    """Return the transposed transition matrix K,
    given rate[i,j] the rate of transition i -> j"""
    n = max([max(v) for v in rate.keys()])
    K = np.zeros((n,n))
    for (i,j), r in rate.items():
        K[j-1,i-1] = r
    for i in range(n):
        K[i,i] = -np.sum(K[:,i])
    return K

def twostate(a, b):
    """
    Basic two-state promoter.

    Parameters
    ----------
    a : positive float
        Rate of transition 2 -> 1.
    b : positive float
        Rate of transition 1 -> 2.

    Returns
    -------
    out : dict
        Transition rates in the form {(i,j): rate[i->j]}.
    """
    if np.min([a,b]) <= 0:
        raise ValueError('a and b must be positive')
    return {(1,2): b, (2,1): a}

def cyclic(a, b=None):
    """
    Cyclic promoter with three or more states.

    Parameters
    ----------
    a : list of positive floats
        The length n of a is the number of states.
        Its i-th term is the rate of transition i -> i+1 (or n -> 1).
    b : list of nonnegative floats, optional
        It must be of same size as a.
        Its i-th term is the rate of transition i+1 -> i (or 1 -> n).

    Returns
    -------
    out : dict
        Transition rates in the form {(i,j): rate[i->j]}.
    """
    n = np.size(a) # Number of states
    rate = {}
    ### Detect potential problems
    if (n == 2) and (b is None):
        return twostate(a[1], a[0])
    if np.min(a) <= 0:
        raise ValueError('rates of a must be positive')
    if b is not None:
        if n < 3:
            raise ValueError('size of a must be 3 or more')
        if np.size(b) != n:
            raise ValueError('a and b must be of same size')
        elif np.min(b)<0:
            raise ValueError('rates of b must be nonnegative')
    else:
        b = np.zeros(n)
    ### Fill the rate dictionary
    for i in range(1,n+1):
        j = (i % n) + 1
        k = (i % n) - 1
        rate[i,j] = a[k]
        if b[k] > 0:
            rate[j,i] = b[k]
    return rate

def dirichlet(a):
    """
    Special promoter leading to Dirichlet distribution.

    Parameters
    ----------
    a : list of positive floats
        The length n of a is the number of states.
        Its i-th term is the rate of all transitions j -> i.

    Returns
    -------
    out : dict
        Transition rates in the form {(i,j): rate[i->j]}.
    """
    n = np.size(a) # Number of states
    rate = {}
    ### Detect potential problems
    if n < 2:
        raise ValueError('size of a must be 2 or more')
    if np.min(a) <= 0:
        raise ValueError('rates of a must be positive')
    ### Fill the rate dictionary
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i != j:
                rate[j,i] = a[i-1]
    return rate
