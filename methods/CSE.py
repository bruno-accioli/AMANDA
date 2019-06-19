import numpy as np
from sklearn.utils import check_array
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors.kde import KernelDensity
from sklearn import mixture


def _gmm_with_pdf(points, allPoints, numComponents, sampleWeights=None,
                density_model=None):
    if (density_model is None):
        if len(allPoints) < numComponents:
            numComponents = len(allPoints)
        clf = mixture.GaussianMixture(n_components=numComponents,
                                      covariance_type='full')
        clf.fit(allPoints)
    else:
        clf = density_model

    return np.exp(clf.score_samples(points)), clf


def _bayesian_gmm(points, allPoints, numComponents, sampleWeights=None,
                 density_model=None):
    if (density_model is None):
        if len(allPoints) < numComponents:
            numComponents = len(allPoints)
        clf = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=100000, n_components=2 * numComponents,
            reg_covar=0, init_params='random', max_iter=1500,
            mean_precision_prior=.8)
        clf.fit(allPoints)
    else:
        clf = density_model

    return np.exp(clf.score_samples(points)), clf


def _kde(points, allPoints, numComponents=None, sampleWeights=None,
         density_model=None):
    if (density_model is None):
        kernel = KernelDensity(kernel='gaussian', bandwidth=0.4)

        if sampleWeights is None:
            kernel = kernel.fit(allPoints)
        else:
            kernel = kernel.fit(allPoints, sample_weight=sampleWeights)
    else:
        kernel = density_model

    return np.exp(kernel.score_samples(points)), kernel


def _pdf_by_class(X, y, classes, densityFunction, sampleWeights=None,
                  estimate_density_by_class=False):
    indexes_by_class = {c: np.where(y == c)[0] for c in classes}

    pdfs_by_class = {}
    numClasses = len(classes)
    density_model = None

    for c, indexes in indexes_by_class.items():
        if indexes.shape[0] > 0:
            pdfs = np.repeat(-1, X.shape[0])
            class_instances = X[indexes]

            # points from a class, all points, number of components
            if (estimate_density_by_class):
                pdfs_by_points, _ = densityFunction(class_instances, X,
                                                    numClasses, sampleWeights)
            else:
                pdfs_by_points, density_model = densityFunction(class_instances,
                                                                X, numClasses,
                                                                sampleWeights,
                                                                density_model)
            pdfs[indexes] = pdfs_by_points
#            a = 0
#            for i in indexes:
#                if pdfsByPoints[a] != -1:
#                    pdfs[i]=pdfsByPoints[a]
#                a+=1
            pdfs_by_class[c] = pdfs

    return pdfs_by_class


def _get_density_function(density_function):
    supported_density_functions = ['kde', 'GaussianMixture',
                                   'BayesianGaussianMixture']

    if density_function not in supported_density_functions:
        raise ValueError("Amanda supports only weight methods in {}, got {}".
                         format(supported_density_functions,
                                density_function))

    if density_function == 'kde':
        return _kde
    elif density_function == 'BayesianGaussianMixture':
        return _bayesian_gmm
    elif density_function == 'GaussianMixture':
        return _gmm_with_pdf


class CSE():
    def __init__(self, density_function='kde', classes=[],
                 estimate_density_by_class=False):
        self.density_function = _get_density_function(density_function)
        self.classes = classes
        self.estimate_density_by_class = estimate_density_by_class

    def get_core(excluding_percentage, Xt, yt, Xt_1=None, yt_1=None,
                 sample_weights):
        Xt = check_array(Xt, dtype='numeric')
        yt = check_array(yt, ensure_2d=False)

        if Xt_1 is not None:
            Xt_1 = check_array(Xt, dtype='numeric')
            X = np.vstack([Xt, Xt_1])
        else:
            X = Xt

        if yt is not None:
            yt_1 = check_array(yt_1, ensure_2d=False)
            y = np.vstack([yt, yt_1])
        else:
            y = yt

        # pesar intancias
        weights_by_class = _pdf_by_class(X, y, self.classes,
                                         self.density_function, sample_weights,
                                         self.estimate_density_by_class)

        # selecionar instancias
        selected_indexes = _select_instances_indexes(weigths_by_class,
                                                     excluding_percentage)

        if len(selected_indexes) > 0:
            X = X[selected_indexes]
            y = y[selected_indexes]

        return X, y

    def _select_instances_indexes(weigths_by_class, excluding_percentage):
        selected_indexes = []

        for c in weigths_by_class:
            class_instances = weigths_by_class[c]
#            numSelected = int(np.floor(criteria*len(arrPdf[arrPdf>-1])))
            num_selected = int(excluding_percentage * class_instances.shape[0])

            indexes = (-class_instances).argsort()[:num_selected]
            selected_indexes.append(indexes)

        stacked_indexes = selected_indexes[0]
        for i in range(1, len(selected_indexes)):
            stacked_indexes = np.hstack([stacked_indexes, selected_indexes[i]])

        return stacked_indexes
