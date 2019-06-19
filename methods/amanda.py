import math
from abc import abstractmethod
import numpy as np
from scipy.spatial.distance import euclidean
from skmultiflow.core import ClassifierMixin
from skmultiflow.core import BaseSKMObject
from sklearn.base import BaseEstimator
from sklearn.semi_supervised import LabelSpreading
import CSE


def _get_model(classifier):
    supported_classfiers = ['labelspreading']

    if isinstance(classifier, BaseEstimator):
        return classifier

    if classifier not in supported_classfiers:
        raise ValueError("Amanda supports only classifiers in {}, got {}".
                         format(supported_classfiers, classifier))

    if classifier == 'labelspreading':
        return LabelSpreading(kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2,
                              max_iter=30, tol=0.001, n_jobs=None)


def _get_distance(distance, beta=None):
    supported_distances = ['hellinger', 'BBD']

    if distance not in supported_distances:
        raise ValueError("Amanda supports only weight methods in {}, got {}".
                         format(supported_distances, distance))

    if distance == 'Hellinger':
        return cutting_percentage_hellinger

    if distance == 'Hellinger2':
        return cutting_percentage_hellinger2

    if distance == 'BBD':
        _check_valid_beta(beta)
        return cutting_percentage_bbd


def _check_valid_beta(beta):
    if (isinstance(beta, int) or isinstance(beta, float)):
        if (beta >= 0 and beta <= 1):
            raise ValueError("The BBD beta value can't be in the interval "
                             "[0, 1], got {}".format(beta))
    else:
        raise ValueError("The BBD beta value must be a number, got {}: {}".
                         format(type(beta).__name__, beta))


_SQRT2 = np.sqrt(2)
def hellinger(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def BhattacharyyaCefficient(p,q):
    return np.sqrt(np.multiply(p,q)).sum()


def BBD(bc, beta):
    return np.log(1 - np.subtract(1, bc)/beta) / np.log(1 - 1/beta)


def cutting_percentage_hellinger(Xt_1, Xt, t=None):
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

def cutting_percentage_hellinger2(Xt_1, Xt, t=None):
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
        q = np.append(q, [max(1-np.sum(q), 0)])
        res.append(hellinger(p, q))
    
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

def cutting_percentage_bbd(Xt_1, Xt, beta, t=None):
    bcs = []
    NXt_1 = len(Xt_1)    
    NXt = len(Xt)
    bins = int(np.sqrt(NXt_1))    
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]        
        hP = np.histogram(P, bins=bins)
        hQ = np.histogram(Q, bins=hP[1])
        bcs.append(BhattacharyyaCefficient(hP[0] / NXt_1, hQ[0] / NXt))
    
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



class AmandaBase(BaseSKMObject, ClassifierMixin):
    def __init__(self, classifier='labelspreading',
                 instance_weight_method='kde', reset=True,
                 estimate_density_by_class=False):

        self.classifier = classifier
        self.cse_method = CSE(density_function='kde',
                              estimate_density_by_class=False)
        self.reset = reset
        self.current_batch = None
        self.current_batch_predictions = None
        self.last_batch_X = None
        self.last_batch_y = None

    def fit(self, X, y, classes=None, sample_weight=None):
        self.model = _get_model(self.classifier)
        self.classes = list(set(y))
        self.cse_method = self.classes

        self.model.fit(X, y)

        self._last_batch_X = X
        self._last_batch_y = y

        self._first_train = True

        return self

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):

        if (self._first_train is False):
            self.fit(self._last_batch_X, self._last_batch_y)

        self.current_batch = X
        self.current_batch_predictions = self.model.predict(X)

        # calculate alpha
        cutting_percentage = self._get_cutting_percentage()

        # CSE
#        if (self.reset):
#            all_instances = X
#            all_labels = self.predicted
#        else:
#            all_instances = np.vstack([self.last_batch_X, X])
#            all_labels = np.hstack([self.last_batch_y,
#                                    self.current_batch_predictions])
        # CSE
        if (self.reset):
            core_instances, core_labels =\
                self.cse_method.get_core(cutting_percentage, X,
                                         self.current_batch_predictions)
        else:
            core_instances, core_labels =\
                self.cse_method.get_core(cutting_percentage, X,
                                         self.current_batch_predictions,
                                         self.last_batch_X, self.last_batch_y)

        self.last_batch_X = core_instances
        self.last_batch_y = core_labels

        self._first_train = False

        return self

    def predict(self, X):
        if (np.array_equal(self.current_batch, X)):
            return self.predicted
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        super().predict_proba(X)

    @abstractmethod
    def _get_cutting_percentage(self):
        raise NotImplementedError


class AmandaFCP(AmandaBase):
    def __init__(self, excluding_percentage=0.5, classifier='labelspreading',
                 instance_weight_method='kde', reset=True,
                 estimate_density_by_class=False):

        super().__init__(classifier, instance_weight_method, reset,
                         estimate_density_by_class)
        self.excluding_percentage = excluding_percentage
        self.excluding_percentage_list = []

    def _get_cutting_percentage(self):
        return self.excluding_percentage


class AmandaDCP(AmandaBase):
    def __init__(self, classifier='labelspreading',
                 instance_weight_method='kde', distance='Hellinger',
                 beta=None, reset=True, estimate_density_by_class=False):

        super().__init__(classifier, instance_weight_method, reset,
                         estimate_density_by_class)
        self.distance_method = _get_distance(distance, beta)
        self.beta = beta

    # TO DO
    def _get_cutting_percentage(self):
        return super()._get_excluding_percentage()
