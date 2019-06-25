import numpy as np
from source import metrics
from source import util
from source import classifiers
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import euclidean
from sklearn.metrics import f1_score
import math



def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]


def makeAccuracy(arrAllAcc, arrTrueY):
    arrAcc = []
    ini = 0
    end = ini
    for predicted in arrAllAcc:
        predicted = np.asarray(predicted)
        predicted = predicted.flatten()
        batchSize = len(predicted)
        ini=end
        end=end+batchSize

        yt = arrTrueY[ini:end]
        arrAcc.append(metrics.evaluate(yt, predicted))
        
    return arrAcc


_SQRT2 = np.sqrt(2)
def hellinger(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2

def BC(p,q):
    return np.sqrt(np.multiply(p,q)).sum()

def BBD(bc, beta):
    return np.log(1 - np.subtract(1,bc)/beta) / np.log(1 - 1/beta)

def cuttingPercentage(Xt_1, Xt, distanceMetric, epsilons=[], hds=[], alpha=None, 
                      beta=None, t=None):
    if distanceMetric == 'Hellinger':
        return cuttingPercentageHellinger(Xt_1, Xt, t)
    
    if distanceMetric == 'Hellinger2':
        return cuttingPercentageHellinger2(Xt_1, Xt, t)
    
    if distanceMetric == 'BBD':
        return cuttingPercentageBBD(Xt_1, Xt, beta, t)
    
    if distanceMetric == 'HDDDM':
        return cuttingPercentageHDDDM(Xt_1, Xt, epsilons, hds, alpha)
    
    return ValueError("""Supported Distance Metrics are ['BBD', 'Hellinger', 'Hellinger2', 'HDDDM']. 
                      Received distanceMetric = {}""".format(distanceMetric))

def cuttingPercentageHellinger(Xt_1, Xt, t=None):
    res = []
    
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        bins = int(np.sqrt(len(Xt_1)))
        hP = np.histogram(P+(-np.min(P)), bins=bins)
        hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
        
        #prevent negative bins       
        if((hP[1]<0).any() or (hQ[1]<0).any()):
            minimum = np.min([hP[1].min(), hQ[1].min()])
            res.append(hellinger(hP[1]-minimum, hQ[1]-minimum))
        else:
            res.append(hellinger(hP[1], hQ[1]))
    
    H = np.mean(res)
    alpha = _SQRT2-H
    #print(t, H, alpha)
    #if alpha < 0:
    #    alpha *= -1
    
    # Sanity Check
#    validation = [True if (i>1 or i<0) else False for i in res]
#    filtered = [i for (i, v) in zip(res, validation) if v]
#    
#    if True in validation:
#        warnings.warn("t={} : Hellinger1 Invalid distance value(s): {}".format(t,filtered))
#        
#    if (alpha > 1 or alpha < 0):
#        warnings.warn("t={} : Hellinger1 Invalid calculated alpha value: {}".format(t,alpha))
    
    if alpha > 0.9:
        alpha = 0.9
    elif alpha < 0.5:
        alpha = 0.5
    return 1-alpha #percentage of similarity

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
    alpha = 1-H
    #print(t, H, alpha)
    #if alpha < 0:
    #    alpha *= -1
    
    # Sanity Check
#    validation = [True if (i>1 or i<0) else False for i in res]
#    filtered = [i for (i, v) in zip(res, validation) if v]
#    
#    if True in validation:
#        warnings.warn("t={} : Hellinger2 Invalid distance value(s): {}".format(t,filtered))
#        
#    if (alpha > 1 or alpha < 0):
#        warnings.warn("t={} : Hellinger2 Invalid calculated alpha value: {}".format(t,H))        
#    
    if alpha > 0.9:
        alpha = 0.9
    elif alpha < 0.5:
        alpha = 0.5
    return 1-alpha #percentage of similarity

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
    alpha = 1-b
    #print(t, H, alpha)
    #if alpha < 0:
    #    alpha *= -1
    
    # Sanity Check
#    validation = [True if (i>1 or i<0) else False for i in bcs]
#    filtered = [i for (i, v) in zip(bcs, validation) if v]
#    
#    if True in validation:
#        warnings.warn("t={} : BBD Invalid Bhatacheryya Coefficient value(s): {}".format(t,filtered))
#        
#    if (alpha > 1 or alpha < 0):
#        warnings.warn("t={} : BBD Invalid calculated alpha value: {}".format(t,b))        
    
    if alpha > 0.9:
        alpha = 0.9
    elif alpha < 0.5:
        alpha = 0.5
    return 1-alpha #percentage of similarity

def cuttingPercentageHDDDM(Xt_1, Xt, epsilons, hds, alpha, k=0.1, gamma=1, t=None):
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
    if hds:
        episilon = abs(H-hds[-1])
        epsilons.append(episilon)
        hds.append(H)
        
        episilon_mean = np.mean(epsilons)
        episilon_std = np.std(epsilons)
        
        beta = episilon_mean + gamma * episilon_std
    else:
        hds.append(H)
        episilon = 0
        beta = np.inf
    
    alpha = 1-alpha
    if (episilon > beta): #drift happened
        alpha = 0.9
        hds = [hds[-1]]
        epsilons = []
    else:
        alpha = alpha - k
        if alpha < 0.5:
            alpha = 0.5
    
    
    #alpha = 1-H
    #print(t, H, alpha)
    #if alpha < 0:
    #    alpha *= -1
        
#    if alpha > 0.9:
#        alpha = 0.9
#    elif alpha < 0.5:
#        alpha = 0.5
    return 1-alpha, hds, epsilons


def cuttingPercentage3(Xt_1, Xt, t=None):
    res = []
    reset = False
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        bins = int(np.sqrt(len(Xt_1)))
        hP = np.histogram(P+(-np.min(P)), bins=bins)
        hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
        res.append(hellinger(hP[1], hQ[1]))

    H = np.mean(res)
    lowerBound = np.power(H, 2)
    upperBound = np.sqrt(2)*H

    similarity = 1-H/upperBound #1 - (((100 * res)/x)/100)#(100 - ((100 * res)/x))/100
    middle = abs(upperBound - lowerBound)
    #print(t, H, lowerBound, middle, similarity)
      
    if lowerBound > upperBound:
        #print(t, res, similarity)
        similarity = abs(middle-H)
        reset = True
    else:
        similarity = H
        reset = False

    #similarity = 0.5+((H / upperBound))
    
    if similarity > 0.9:
        similarity = 0.9
    elif similarity < 0.5:
        similarity = 0.5
    
    #print("step {}, similarity = {}, reset = {} ".format(t, similarity, reset))
    return similarity, reset #percentage of similarity


def cuttingPercentage2(Xt_1, Xt, t=None):
    res = []
    reset = False
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        bins = int(np.sqrt(len(Xt_1)))
        hP = np.histogram(P+(-np.min(P)), bins=bins)
        hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
        res.append(hellinger(hP[1], hQ[1]))
    res = np.mean(res)
    x = np.sqrt(2)

    #similarity = abs((100 - ((100 * res)/x))/100) #ensure non-negativity
    similarity = 1 - (((100 * res)/x)/100)#(100 - ((100 * res)/x))/100
    #print(t, res, similarity)
      
    if similarity < 0:
        #print(t, res, similarity)
        reset = True
    elif similarity > 0:
        reset = False

    similarity = 0.5+((res / x)/10)
    if similarity > 0.9:
        similarity = 0.9
    
    #print(similarity)
    
    return similarity, reset #percentage of similarity


class run(BaseEstimator, ClassifierMixin):

    def __init__(self, K=1, sizeOfBatch=100, batches=50, poolSize=100, isBatchMode=True, initialLabeledData=50, clfName='lp', distanceMetric='Hellinger', beta=None):
        self.sizeOfBatch = sizeOfBatch
        self.batches = batches
        self.initialLabeledData=initialLabeledData
        self.usePCA=False
        self.distanceMetric = distanceMetric
        #used only by gmm and cluster-label process
        self.densityFunction='kde'
        self.K = K
        self.clfName = clfName.lower()
        self.poolSize = poolSize
        self.isBatchMode = isBatchMode
        
        #used only by BBD distance metric
        self.beta = beta
        
        #print("{} excluding percecntage".format(excludingPercentage))    
    
    def get_params(self, deep=True):
        return {"K":self.K, "sizeOfBatch":self.sizeOfBatch, "batches":self.batches, 
                "poolSize":self.poolSize, "isBatchMode":self.isBatchMode, 
                "clfName":self.clfName, "distanceMetric":self.distanceMetric,
                "beta":self.beta}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
            
    def fit(self, dataValues, dataLabels=None):
        arrAcc = []
        arrF1 = []
        classes = list(set(dataLabels))
        initialDataLength = 0
        finalDataLength = self.initialLabeledData
        excludingPercentage = 0.5
        epsilons = [] 
        hds = []

        # ***** Box 1 *****
        #Initial labeled data
        X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
        reset = True
        if self.isBatchMode:
            for t in range(self.batches):
                #print("passo: ",t)
                initialDataLength=finalDataLength
                finalDataLength=finalDataLength+self.sizeOfBatch
                # ***** Box 2 *****            
                Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
                
                # ***** Box 3 *****
                clf = classifiers.classifier(X, y, self.K, self.clfName)
                predicted = clf.predict(Ut)
                # Evaluating classification
                arrAcc.append(metrics.evaluate(yt, predicted))
                arrF1.append(f1_score(yt, predicted, average='macro'))

                # ***** Box 4 *****
                #excludingPercentage = cuttingPercentage(X, Ut, t)
                if self.distanceMetric != 'HDDDM':
                    excludingPercentage = cuttingPercentage(X, Ut, self.distanceMetric, 
                                                            epsilons, hds, 
                                                            excludingPercentage, self.beta, t)
                else:
                    excludingPercentage, hds, epsilons = cuttingPercentage(X, Ut, self.distanceMetric, 
                                                                            epsilons, hds, 
                                                                            excludingPercentage, self.beta, t)
                allInstances = []
                allLabels = []
                
                # ***** Box 5 *****
                if reset == True:
                    #Considers only the last distribution (time-series like)
                    pdfsByClass = util.pdfByClass(Ut, yt, classes, self.densityFunction)
                else:
                    #Considers the past and actual data (concept-drift like)
                    allInstances = np.vstack([X, Ut])
                    allLabels = np.hstack([y, yt])
                    pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, self.densityFunction)
                    
                selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)
            
                # ***** Box 6 *****
                if reset == True:
                    #Considers only the last distribution (time-series like)
                    X, y = util.selectedSlicedData(Ut, yt, selectedIndexes)
                else:
                    #Considers the past and actual data (concept-drift like)
                    X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)
        else:
            t=0
            inst = []
            labels = []
            clf = classifiers.classifier(X, y, self.K, self.clfName)
            remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), self.usePCA)
            reset = True
            for Ut, yt in zip(remainingX, remainingY):
                allInstances = []
                allLabels = []
                predicted = clf.predict(Ut.reshape(1, -1))
                arrAcc.append(predicted[0])
                inst.append(Ut)
                labels.append(predicted[0])
                
                if len(inst) == self.poolSize:
                    inst = np.asarray(inst)
                    excludingPercentage = cuttingPercentage(X, inst, t)
                    t+=1
                    if reset == True:
                        #Considers only the last distribution (time-series like)
                        pdfsByClass = util.pdfByClass(inst, labels, classes, self.densityFunction)
                    else:
                        #Considers the past and actual data (concept-drift like)
                        allInstances = np.vstack([X, inst])
                        allLabels = np.hstack([y, labels])
                        pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, self.densityFunction)
                    
                    selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

                    if reset == True:
                        #Considers only the last distribution (time-series like)
                        X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                    else:
                        #Considers the past and actual data (concept-drift like)
                        X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)

                    clf = classifiers.classifier(X, y, self.K, self.clfName)
                    inst = []
                    labels = []

            arrAcc = split_list(arrAcc, self.batches)
            arrAcc = makeAccuracy(arrAcc, remainingY)   
            
     
        # returns accuracy array and last selected points
        #self.threshold_ = arrAcc
        self.threshold_ = arrF1
        return self
    
    def predict(self):
        try:
            getattr(self, "threshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return self.threshold_
    
    def score(self, X, y=None):
        accuracies = self.predict()
        N = len(accuracies)
        #print(self.K, self.excludingPercentage, sum(accuracies)/N)
        return sum(accuracies)/N
