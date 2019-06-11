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
        res.append(hellinger(hP[0] / NXt_1, hQ[0] / NXt))
    
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

def plotHistograms(x1, x2):
    bins = int(np.sqrt(len(x1)))
    plt.hist(x1, label='x1', bins=bins, alpha=0.8)
    plt.hist(x2, label='x2', bins=bins, alpha=0.8)
    plt.show()
    
x1 = np.random.normal(loc=0.0, scale=1.0, size=500)

x2 = x1 + 0.7


plotHistograms(x1, x2)

cuttingPercentageHellinger(x1.reshape([-1,1]),x2.reshape([-1,1]))
cuttingPercentageHellinger2(x1.reshape([-1,1]),x2.reshape([-1,1]))
cuttingPercentageBBD(x1.reshape([-1,1]), x2.reshape([-1,1]), beta=-1)