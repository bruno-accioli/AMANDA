import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


_SQRT2 = np.sqrt(2)


def hellinger(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def BC(p, q):
    return np.sqrt(np.multiply(p, q)).sum()


def BBD(bc, beta):
    if beta == 10:
        return -10 * math.log((9 + bc)/10, 2.867971990792441)
    else:
        return np.log(1 - np.subtract(1,bc)/beta) / np.log(1 - 1/beta)


def cuttingPercentageHellinger(Xt_1, Xt, t=None):
    res = []

    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        bins = int(np.sqrt(len(Xt_1)))
        hP = np.histogram(P+(-np.min(P)), bins=bins)
        hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
        
        if((hP[1]<0).any() or (hQ[1]<0).any()):
            minimum = np.min([hP[1].min(), hQ[1].min()])
            res.append(hellinger(hP[1]-minimum, hQ[1]-minimum))
        else:
            res.append(hellinger(hP[1], hQ[1]))
    
    H = np.mean(res)

    return H #percentage of similarity

def cuttingPercentageHellinger2(Xt_1, Xt, t=None):
    res = []
    NXt_1 = len(Xt_1)    
    NXt = len(Xt)    
    bins = int(np.sqrt(NXt_1)) 
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        hP = np.histogram(P, bins=bins)
        hQ = np.histogram(Q, bins=hP[1])
        
        p = np.append(hP[0] / NXt_1, [0])
        q = hQ[0] / NXt
        q = np.append(q, [1-np.sum(q)])
        res.append(hellinger(p, q))
    
    H = np.mean(res)

    return H

def cuttingPercentageBBD(Xt_1, Xt, beta, t=None):
    bcs = []
    NXt_1 = len(Xt_1)    
    NXt = len(Xt)
    bins = int(np.sqrt(NXt_1))    
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]        
        hP = np.histogram(P, bins=bins)
        hQ = np.histogram(Q, bins=hP[1])
        bcs.append(BC(hP[0] / NXt_1, hQ[0] / NXt))
    
    bc = np.mean(bcs)
    b = BBD(bc, beta)

    return b #percentage of similarity    

def plotHistograms(x1, x2, n=1):
    bins = int(np.sqrt(len(x1)))
    
    plt.figure(figsize=(8*3, 5))
    plt.grid(axis='y', alpha=0.75)
    
    plt.subplot(130 + n)  
    
    plt.xlim(-3, 10)
    
    plt.hist(x1, label='x1', bins=bins, color= 'tab:blue', alpha=0.9, rwidth=0.85)
    plt.hist(x2, label='x2', bins=bins, color= 'tab:green', alpha=0.9, rwidth=0.85)
    
    dist = cuttingPercentageHellinger2(x1.reshape([-1,1]),x2.reshape([-1,1]))
    plt.title('Distância = {:.3f}'.format(dist))
    
x1 = np.random.normal(loc=0.0, scale=1.0, size=250)
bins = int(np.sqrt(len(x1)))

offsets = [0.15, 2, 6]
n = 1

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize=(12, 3))

x2 = x1 + offsets[0]
ax1.set_xlim(-3, 10)    
ax1.grid(axis='y', alpha=0.75)
ax1.hist(x1, label='x1', bins=bins, color= 'tab:blue', alpha=0.9, rwidth=0.85)    
ax1.hist(x2, label='x2', bins=bins, color= 'tab:green', alpha=0.9, rwidth=0.85)

dist = cuttingPercentageHellinger2(x1.reshape([-1,1]),x2.reshape([-1,1]))
ax1.set_title('Distância = {:.3f}'.format(dist))

x2 = x1 + offsets[1]
ax2.set_xlim(-3, 10)   
ax2.grid(axis='y', alpha=0.75) 
ax2.hist(x1, label='x1', bins=bins, color= 'tab:blue', alpha=0.9, rwidth=0.85)    
ax2.hist(x2, label='x2', bins=bins, color= 'tab:green', alpha=0.9, rwidth=0.85)

dist = cuttingPercentageHellinger2(x1.reshape([-1,1]),x2.reshape([-1,1]))
ax2.set_title('Distância = {:.3f}'.format(dist))

x2 = x1 + offsets[2]
ax3.set_xlim(-3, 10)    
ax3.grid(axis='y', alpha=0.75)
ax3.hist(x1, label='x1', bins=bins, color= 'tab:blue', alpha=0.8, rwidth=0.7)    
ax3.hist(x2, label='x2', bins=bins, color= 'tab:green', alpha=0.8, rwidth=0.7)

dist = cuttingPercentageHellinger2(x1.reshape([-1,1]),x2.reshape([-1,1]))
ax3.set_title('Distância = {:.3f}'.format(dist))

plt.show() 
#print(cuttingPercentageHellinger(x1.reshape([-1,1]),x2.reshape([-1,1])))
#print(cuttingPercentageHellinger2(x1.reshape([-1,1]),x2.reshape([-1,1])))
#print(cuttingPercentageBBD(x1.reshape([-1,1]), x2.reshape([-1,1]), beta=-1))
#n += 1


#cuttingPercentageHellinger(x1.reshape([-1,1]),x2.reshape([-1,1]))
#
#cuttingPercentageBBD(x1.reshape([-1,1]), x2.reshape([-1,1]), beta=-1)