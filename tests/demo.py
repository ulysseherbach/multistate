### demo.py for the package multistate ###

### For testing
import sys
sys.path.append("../")
### Import the relevant packages
import numpy as np
import matplotlib.pyplot as plt
import multistate as ms

### Paramètres graphiques
options = {'dpi':100, 'bbox_inches':'tight', 'frameon':False}

### Couleurs
rouge = '#D91D12'
orange = '#F06E06'
bleu = '#005CCC'
vert = '#40C027'

### Petit test de promoteur
promoter = {(1,2): 10, (2,3): 4, (3,4): 5, (4,1): 3}
### Bruitage des taux
for key, val in promoter.items():
    promoter[key] += np.random.normal(scale=0.0001)
### Affichage
a, b = ms.refractory.eigenvalues(promoter)
print('a = {}'.format(a) + '\n' + 'b = {}'.format(b))

### Distribution de la période inactive
fig = plt.figure(figsize=(6,2))
x = np.linspace(0,5,1000)
y = ms.refractory.dist_inactive(x, promoter)
plt.plot(x, y, color=rouge, lw=2)
plt.savefig('test_distrib_time.pdf', **options)

### Distribution de l'ARN (Poisson)
fig = plt.figure(figsize=(6,2))
x = np.array(range(41))
y = ms.refractory.dist_poisson(x, promoter, scale=100)
for i in range(np.size(x)):
    plt.plot([x[i],x[i]], [0,y[i]], color=orange)
plt.plot(x, y, color=orange, marker='o', ls='', markersize=3)
plt.savefig('test_dist_poisson.pdf', **options)

### Distribution de l'ARN (PDMP)
fig = plt.figure(figsize=(6,2))
dx = 1e-3
x = 40*np.linspace(dx,1,1000)
y = ms.refractory.dist_pdmp(x, promoter, scale=100)
plt.plot(x, y, color=bleu, lw=2)
plt.savefig('test_dist_pdmp.pdf', **options)

