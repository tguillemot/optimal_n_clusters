# TODO:
# X. Refactor method into specific classes 22/06
# 2. Add verbose method ? 23/06
# . Check if we cannot do something better 23/06
# 3.1. Add warning when n_clusters is 1 for metrics methods
# 3.2. Add a random computation ???
# 4. Add the support of pandas + ... 23/06
# 4.1 Rajouter calcule du meilleur estimateur une fois support de pandas
# X. Add parameters
# X.X Add parameters.range
# X.X Add parameters.specific values
# 5.3 Ajouter plusieurs parametres differents
# 5. Tests 24/06
# 6. Docs 24/06
# 7. Gestion des mauvaise valeurs
# 8. Add parallel

import warnings
import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn.externals import six
from sklearn.utils import check_array, check_random_state
from sklearn.utils.testing import ignore_warnings
from sklearn.model_selection import ParameterGrid


class _NClusterSearchBase(six.with_metaclass(ABCMeta, object)):
    def __init__(self, estimator, parameters):
        self.estimator = estimator
        self.parameters = parameters.copy()

    @abstractmethod
    def _fit_and_score(self, X, y=None):
        pass

    def fit(self, X, y):
        self.score_ = self._fit_and_score(X, y)

        # Ajouter support best_estimator une fois pandas géré
        # optimal_n_clusters = self.n_clusters_range[np.nanargmax(self.score_)]
        # self.best_estimator_ = self.estimator.set_params(
        #     n_clusters=optimal_n_clusters)
        self.best_estimator_ = self.estimator  # remove at the end

        return self


# Search for unsupervised metrics
class UnsupervisedMetricSearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters, n_clusters_range, metric):
        super(UnsupervisedMetricSearch, self).__init__(
            estimator, n_clusters_range)
        self.parameters = parameters
        self.metric = metric

    def _fit_and_score(self, X, y=None):
        parameters_iterable = ParameterGrid(self.parameters)

        score = np.empty(len(parameters_iterable))
        # parameters = np.empty(score.shape, dtype=dict)
        for k, params in enumerate(parameters_iterable):
            try:
                self.estimator.set_params(**params)
                score[k] = self.metric(X, self.estimator.fit_predict(X))
            except ValueError:
                # TODO Add a warning for false values
                warnings.warn('Put a warning.')
                score[k] = np.nan
            # parameters[k] = params

        return score
        # return score, parameters


# Scorer for supervised metrics
class StabilitySearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters, metric, n_draws=10,
                 p_samples=.8, random_state=None):
        super(StabilitySearch, self).__init__(
            estimator, parameters)
        self.metric = metric
        self.n_draws = n_draws
        self.p_samples = p_samples
        self.random_state = random_state

    def _fit_and_score(self, X, y=None):
        n_samples, _ = X.shape
        rng = check_random_state(self.random_state)
        parameters_iterable = ParameterGrid(self.parameters)

        # Compute the random data once for all
        self.data_ = (
            rng.uniform(size=(self.n_draws, 2 * n_samples)) < self.p_samples)

        score = np.empty(len(parameters_iterable))
        # parameters = np.empty(score.shape, dtype=dict)
        for k, params in enumerate(parameters_iterable):
            self.estimator.set_params(**params)
            draw_scores = np.empty(self.n_draws)
            for l, d in enumerate(self.data_):
                p1, p2 = np.split(d, 2)

                labels1 = self.estimator.fit_predict(X[p1])
                labels2 = self.estimator.fit_predict(X[p2])

                draw_scores[l] = self.metric(labels1[p2[p1]], labels2[p1[p2]])
            score[k] = draw_scores.mean()
            # parameters[k] = params

        return score
        # return score, parameters


# Scorer for distortion jump
class DistortionJumpSearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters):
        super(DistortionJumpSearch, self).__init__(
            estimator, parameters)

    def _fit_and_score(self, X, y=None):
        _, n_features = X.shape

        # Extraction of n_clusters or n_components from parameters
        n_clusters_list = self.parameters.pop('n_clusters', None)
        if n_clusters_list is None:
            n_clusters_list = self.parameters.pop('n_components', None)
        if n_clusters_list is None:
            raise ValueError("Pas n_clusters ou n_components")

        # Compute n_clusters parameters
        n_clusters_values, index = np.unique(
            np.hstack((n_clusters_list, np.array(n_clusters_list) + 1)),
            return_index=True)
        index = index < len(n_clusters_list)

        parameters_iterable = ParameterGrid(self.parameters)
        distortion = np.empty((len(n_clusters_values),
                               len(parameters_iterable)))
        # parameters = np.empty(distortion.shape, dtype=dict)
        for k, n_clusters in enumerate(n_clusters_values):
            for l, params in enumerate(parameters_iterable):
                params['n_clusters'] = n_clusters
                self.estimator.set_params(**params)
                distortion[k, l] = (
                    (self.estimator.fit(X).score(X) / n_features) **
                    (-n_features / 2))
                # parameters[k, l] = params

        # return (distortion[index] - distortion[1:][index[:-1]],
        #        parameters[index])
        return distortion[index] - distortion[1:][index[:-1]]


# Scorer for Pham
class PhamSearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters):
        super(PhamSearch, self).__init__(
            estimator, parameters)

    def _fit_and_score(self, X, y=None):
        _, n_features = X.shape

        # Extraction of n_clusters or n_components from parameters
        n_clusters_list = self.parameters.pop('n_clusters', None)
        if n_clusters_list is None:
            n_clusters_list = self.parameters.pop('n_components', None)
        if n_clusters_list is None:
            raise ValueError("Pas n_clusters ou n_components")

        # Compute n_clusters parameters
        n_clusters_values, index = np.unique(
            np.hstack((n_clusters_list, np.array(n_clusters_list) - 1)),
            return_index=True)
        index = index < len(n_clusters_list)

        parameters_iterable = ParameterGrid(self.parameters)
        inertia = np.empty((len(n_clusters_values), len(parameters_iterable)))
        for k, n_clusters in enumerate(n_clusters_values):
            for l, params in enumerate(parameters_iterable):
                if n_clusters == 0:
                    continue
                params['n_clusters'] = n_clusters
                self.estimator.set_params(**params)
                inertia[k, l] = self.estimator.fit(X).score(X)

        weights = 1. - np.exp((n_clusters_values[index] - 2) *
                              np.log(5. / 6.) + np.log(.75) -
                              np.log(n_features))

        with ignore_warnings(category=RuntimeWarning):
            score = np.where(inertia[index] != 0., inertia[index] / (
                weights[:, np.newaxis] * inertia[:-1][index[1:]]), 1.)

        if n_clusters_values[0] == 0:
            score[0, :] = 1.

        return -score


# Scorer for Gap
class GapSearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters, n_draws=10, random_state=None):
        super(GapSearch, self).__init__(estimator, parameters)
        self.n_draws = n_draws
        self.random_state = random_state

    def _fit_and_score(self, X, y=None):
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)

        # Compute the random data once for all
        bb_min, bb_max = np.min(X, 0), np.max(X, 0)
        data = (rng.uniform(size=(self.n_draws, n_samples, n_features)) *
                (bb_max - bb_min) + bb_min)

        # Extraction of n_clusters or n_components from parameters
        n_clusters_list = self.parameters.pop('n_clusters', None)
        if n_clusters_list is None:
            n_clusters_list = self.parameters.pop('n_components', None)
        if n_clusters_list is None:
            raise ValueError("Pas n_clusters ou n_components")

        # Compute n_clusters parameters
        n_clusters_values, index = np.unique(
            np.hstack((n_clusters_list, np.array(n_clusters_list) + 1)),
            return_index=True)
        index = index < len(n_clusters_list)

        parameters_iterable = ParameterGrid(self.parameters)
        expected_inertia = np.empty((len(n_clusters_values),
                                     len(parameters_iterable)))
        estimated_inertia = np.empty(expected_inertia.shape)
        safety = np.empty(expected_inertia.shape)
        inertia_n_draws = np.empty(self.n_draws)

        for k, n_clusters in enumerate(n_clusters_values):
            for l, params in enumerate(parameters_iterable):
                params['n_clusters'] = n_clusters
                self.estimator.set_params(**params)

                estimated_inertia[k, l] = -estimator.fit(X).score(X)
                for t, Xt in enumerate(data):
                    inertia_n_draws[t] = -estimator.fit(Xt).score(Xt)
                inertia_n_draws = np.log(inertia_n_draws)
                expected_inertia[k, l] = np.mean(inertia_n_draws)
                safety[k, l] = (np.std(inertia_n_draws) *
                                np.sqrt(1. + 1. / self.n_draws))

        # Compute the gap score
        gap = expected_inertia - np.log(estimated_inertia)

        return gap[index] - gap[1:][index[:-1]] + safety[1:][index[:-1]]


# Ajouter les arguments qui remplacent
class OptimalNClusterSearch():
    def __init__(self, estimator, parameters, fitting_process='auto',
                 **kwargs):
        self.estimator = estimator
        self.parameters = parameters.copy()
        self.fitting_process = fitting_process
        self.kwargs = kwargs

    def _check_initial_parameters(self, X):
        n_samples, _ = X.shape

        X = check_array(X, dtype=[np.float64, np.float32])
        # if self.n_clusters_range.max() > n_samples:
        #     raise ValueError("Put a message")

    def _select_fitting_method(self):
        fitting_method = {
            'auto': PhamSearch,
            'pham': PhamSearch,
            'gap': GapSearch,
            'distortion_jump': DistortionJumpSearch,
            'stability': StabilitySearch,
            'unsupervised': UnsupervisedMetricSearch
        }

        self.scorer_ = fitting_method[self.fitting_process](
            estimator=self.estimator, parameters=self.parameters,
            **self.kwargs)

    def fit(self, X, y=None):
        self._check_initial_parameters(X)

        self._select_fitting_method()
        self.scorer_.fit(X, y)

        self.score_ = self.scorer_.score_
        self.best_estimator_ = self.scorer_.best_estimator_

        return self


if __name__ == '__main__':
    from sklearn.metrics import calinski_harabaz_score, fowlkes_mallows_score
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs, load_iris

    import matplotlib.pyplot as plt

    n_samples, random_state = 1500, 1
    X1, y1 = make_blobs(n_samples=n_samples, random_state=random_state,
                        centers=3)
    X2, y2 = make_blobs(n_samples=n_samples, random_state=random_state,
                        centers=2)
    X = np.vstack((X1, X2 + 10))
    # X1, y1 = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    # X2, y2 = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    # X = X1  # np.vstack((X1, X2 * 4))

    # iris = load_iris()
    # X = iris.data

    estimator = KMeans()

    n_clusters_range = np.arange(1, 8)
    parameters = {'n_clusters': (1, 2, 3, 4, 6, 7), 'random_state': np.arange(3)}
    search = OptimalNClusterSearch(estimator=estimator,
                                   parameters=parameters,
                                   fitting_process='distortion_jump').fit(X)

    print("Optimal number of clusters : %s" % search.best_estimator_)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])

    print(search.score_)

    plt.figure()
    plt.plot(parameters['n_clusters'], search.score_, 'o-', alpha=.6)
    plt.show()
