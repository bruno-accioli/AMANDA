import numpy as np
import math
from source import classifiers
from source import metrics
from source import util
from scipy.spatial.distance import euclidean
import warnings


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
    if beta == 10:
        return -10 * math.log((9 + bc)/10, 2.867971990792441)
    else:
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
    
    return ValueError("""Supported Distance Metrics are ['BBD', 'Hellinger', 'Hellinger2']. 
                      Received distanceMetric = {}""".format(distanceMetric))
    
def cuttingPercentageHellinger(Xt_1, Xt, t=None):
    res = []
    
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        bins = int(np.sqrt(len(Xt_1)))
        hP = np.histogram(P+(-np.min(P)), bins=bins)
        hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
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
    return 1-similarity, reset #percentage of dissimilarity


def cuttingPercentageByClass(Xt_1, Xt, yt_1, yt, classes, t=None):
    x = np.sqrt(2)
    reset = False

    hellinger_distance_by_class = {}
    similarityByClass = {}

    indexes_Xt_1_ByClass = util.slicingClusteredData(yt_1, classes)
    indexes_Xt_ByClass = util.slicingClusteredData(yt, classes)    

    for c in classes:
        res = []
        for i in range(Xt_1.shape[1]):
            P = Xt_1[indexes_Xt_1_ByClass[c], i]
            Q = Xt[indexes_Xt_ByClass[c], i]

            bins = int(np.sqrt(len(indexes_Xt_1_ByClass[c])))

            hP = np.histogram(P+(-np.min(P)), bins=bins)
            hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
            res.append(hellinger(hP[1], hQ[1]))

        res = np.mean(res)
        similarity = 1 - (((100 * res)/x)/100)#(100 - ((100 * res)/x))/100
        #print(t,res, similarity)
        if similarity < 0:
            reset = True
        elif similarity > 0:
            reset = False

        similarity = 0.5+((res / x)/10)
        if similarity > 0.9:
            similarity = 0.9

        similarityByClass.update({c: similarity})
        #print(t,c,similarity)

    return similarityByClass, reset #percentage of similarity


def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledData = kwargs["initialLabeledData"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    classes = kwargs["classes"]
    K = kwargs["K_variation"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    clfName = kwargs["clfName"]
    densityFunction = kwargs["densityFunction"]
    poolSize = kwargs["poolSize"]
    isBatchMode = kwargs["isBatchMode"]
    distanceMetric = kwargs['distanceMetric']
    
    if distanceMetric == 'BBD':
        beta = kwargs["beta"]
    else:
        beta = None
    
    print("METHOD: {} as classifier and {} and {} distance as dynamic CSE".format(clfName, densityFunction, distanceMetric))
    usePCA=False
    arrAlphas = []
    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    arrPredicted = []
    epsilons = [] 
    hds = []
    excludingPercentage = 0.5
    initialDataLength = 0
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    reset = True
    if isBatchMode:
        for t in range(batches):
            #print("passo: ",t)
            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+sizeOfBatch
            #print(initialDataLength)
            #print(finalDataLength)
            # ***** Box 2 *****
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
            
            # ***** Box 3 *****
            clf = classifiers.classifier(X, y, K, clfName)#O(nd+kn)

            # for decision boundaries plot
            arrClf.append(clf)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(np.array(Ut))
            arrYt.append(yt)
            predicted = clf.predict(Ut)
            arrPredicted.append(predicted)
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))
            
            # ***** Box 4 *****
            if distanceMetric != 'HDDDM':
                excludingPercentage = cuttingPercentage(X, Ut, distanceMetric, 
                                                        epsilons, hds, 
                                                        excludingPercentage, beta, t)
            else:
                excludingPercentage, hds, epsilons = cuttingPercentage(X, Ut, distanceMetric, 
                                                        epsilons, hds, 
                                                        excludingPercentage, beta, t)
            arrAlphas.append(excludingPercentage)
            #excludingPercentageByClass, reset = cuttingPercentageByClass(X, Ut, y, predicted, classes, t)
            allInstances = []
            allLabels = []
            
            # ***** Box 5 *****
            if reset == True:
                #Considers only the last distribution (time-series like)
                pdfsByClass = util.pdfByClass(Ut, predicted, classes, densityFunction)#O(n^{2}d)
            else:
                #Considers the past and actual data (concept-drift like)
                allInstances = np.vstack([X, Ut])
                allLabels = np.hstack([y, predicted])
                pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, densityFunction)
                
            selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)#O(n log(n) c)
            #selectedIndexes = util.compactingDataDensityBased(pdfsByClass, excludingPercentageByClass)
            #print(t, excludingPercentage)
            # ***** Box 6 *****
            if reset == True:
                #Considers only the last distribution (time-series like)
                X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)#O(n)
            else:
                #Considers the past and actual data (concept-drift like)
                X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)
    else:
        t=0
        inst = []
        labels = []
        clf = classifiers.classifier(X, y, K, clfName)
        remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
        reset = False
        for Ut, yt in zip(remainingX, remainingY):
            predicted = clf.predict(Ut.reshape(1, -1))[0]
            arrAcc.append(predicted)
            inst.append(Ut)
            labels.append(predicted)

            # for decision boundaries plot
            arrClf.append(clf)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(Ut)
            arrYt.append(yt)
            
            arrPredicted.append(predicted)
            
            #new approach
            if len(inst) == poolSize:
                inst = np.array(inst)
                excludingPercentage = cuttingPercentage(X, inst, t)
                t+=1
                '''if excludingPercentage < 0:
                    #print("negative, reseting points")
                    excludingPercentage = 0.5 #default
                    reset = True
                else:
                    reset = False
                '''
                if reset == True:
                    #Considers only the last distribution (time-series like)
                    pdfsByClass = util.pdfByClass(inst, labels, classes, densityFunction)
                else:
                    #Considers the past and actual data (concept-drift like)
                    allInstances = np.vstack([X, inst])
                    allLabels = np.hstack([y, labels])
                    pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, densityFunction)
                
                selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

                if reset == True:
                    #Considers only the last distribution (time-series like)
                    X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                else:
                    #Considers the past and actual data (concept-drift like)
                    X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)

                clf = classifiers.classifier(X, y, K, clfName)
                inst = []
                labels = []
            
        arrAcc = split_list(arrAcc, batches)
        arrAcc = makeAccuracy(arrAcc, remainingY)
        arrYt = split_list(arrYt, batches)
        arrPredicted = split_list(arrPredicted, batches)

    # returns accuracy array and last selected points
    if distanceMetric == 'BBD':
        clfMethod = "AMANDA-DCP {} - \u03B2 = {:.3f}".format(distanceMetric, beta)
    else:
        clfMethod = "AMANDA-DCP {}".format(distanceMetric)
    
    print("{} | Mean keeping percentage={:.2f} | Std keeping percentage={:.2f}".format(clfMethod, 
          np.mean(arrAlphas), 
          np.std(arrAlphas)))  
    
#    print(arrAlphas)
      
    return clfMethod, arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted