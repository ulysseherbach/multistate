### Small demo for the multistate package ###
import numpy as np
import matplotlib.pyplot as plt
import multistate as ms

### Graphical parameters
options = {'dpi': 100, 'bbox_inches': 'tight'}
### Colors
rouge = '#D91D12'
orange = '#F06E06'
bleu = '#005CCC'
vert = '#40C027'

### Little promoter test
promoter = {(1,2): 10, (2,3): 4, (3,4): 5, (4,1): 3}
### Jitter the rates (optional)
for key, val in promoter.items():
    promoter[key] += np.random.normal(scale=0.0001)
### Promoter eigenvalues
a, b = ms.refractory.eigenvalues(promoter)
print(f"a = {a}" + "\n" + f"b = {b}")

### Distribution of the inactive period
fig = plt.figure(figsize=(6,2))
x = np.linspace(0,5,1000)
y = ms.refractory.dist_inactive(x, promoter)
plt.plot(x, y, color=rouge, lw=2)
plt.savefig("../results/test_distrib_time.pdf", **options)

### Distribution of RNA (Poisson-mixture)
fig = plt.figure(figsize=(6,2))
x = np.array(range(41))
y = ms.refractory.dist_poisson(x, promoter, scale=100)
for i in range(np.size(x)):
    plt.plot([x[i],x[i]], [0,y[i]], color=orange)
plt.plot(x, y, color=orange, marker='o', ls='', markersize=3)
plt.savefig("../results/test_dist_poisson.pdf", **options)

### Distribution of RNA (PDMP)
fig = plt.figure(figsize=(6,2))
dx = 1e-3
x = 40*np.linspace(dx,1,1000)
y = ms.refractory.dist_pdmp(x, promoter, scale=100)
plt.plot(x, y, color=bleu, lw=2)
plt.savefig("../results/test_dist_pdmp.pdf", **options)

### Simulation of trajectories
u = 100*np.array([1,0,0,0])
Tmax = 20
### PDMP
fig = plt.figure(figsize=(6,2))
T = np.linspace(0,Tmax,1000)
traj = ms.sim_pdmp(promoter, T)
X = traj['X'][:,0]
plt.plot(T, X, color=bleu)
plt.savefig("../results/test_traj_pdmp.pdf", **options)
### Classic
fig = plt.figure(figsize=(6,2))
traj = ms.sim_ssa(promoter, u, Tmax)
T, E, M = traj['T'], traj['E'], traj['M']
jtraj = ms.simulation.get_jtraj(T, E) # Minimal promoter path
plt.step(T, M, color=orange)
plt.savefig("../results/test_traj_ssa.pdf", **options)
### Conditional PDMP
fig = plt.figure(figsize=(6,2))
T = np.linspace(0,Tmax,1000)
x0 = np.array([0,0,0,1])
ctraj = ms.conditional_pdmp(T, x0, jtraj)
X = np.dot(ctraj['X'], u)
plt.plot(T, X, color=bleu)
plt.savefig("../results/test_traj_cond_pdmp.pdf", **options)
