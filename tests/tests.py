### Main tests for the multistate package ###
import numpy as np
import multistate as ms

### Little promoter test
promoter = {(1,2): 10, (2,3): 4, (3,4): 5, (4,1): 3}

### Jitter the rates (optional)
for key, val in promoter.items():
    promoter[key] += np.random.normal(scale=0.0001)

### Promoter eigenvalues
a, b = ms.refractory.eigenvalues(promoter)
print('a = {}'.format(a) + '\n' + 'b = {}'.format(b))

### Distribution of the inactive period
x = np.linspace(0,5,1000)
y = ms.refractory.dist_inactive(x, promoter)

### Distribution of RNA (Poisson-mixture)
x = np.array(range(41))
y = ms.refractory.dist_poisson(x, promoter, scale=100)

### Distribution of RNA (PDMP)
dx = 1e-3
x = 40*np.linspace(dx,1,1000)
y = ms.refractory.dist_pdmp(x, promoter, scale=100)

### Simulation of trajectories
u = 100*np.array([1,0,0,0])
Tmax = 20
### PDMP
T = np.linspace(0,Tmax,1000)
traj = ms.sim_pdmp(promoter, T)
X = traj['X'][:,0]
### Classic
traj = ms.sim_ssa(promoter, u, Tmax)
T, E, M = traj['T'], traj['E'], traj['M']
jtraj = ms.simulation.get_jtraj(T, E) # Minimal promoter path
### Conditional PDMP
T = np.linspace(0,Tmax,1000)
x0 = np.array([0,0,0,1])
ctraj = ms.conditional_pdmp(T, x0, jtraj)
X = np.dot(ctraj['X'], u)
