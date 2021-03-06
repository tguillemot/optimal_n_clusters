# TODO:
# X. Refactor method into specific classes 22/06
# 2. Add verbose method ? 23/06
# X Check if we cannot do something better 23/06
# 3.1. Add warning when n_clusters is 1 for metrics methods
# 3.2. Add a random computation ???
# X. Add the support of pandas + ... 23/06
# X.X Rajouter calcul du meilleur estimateur une fois support de pandas
# X.X Corriger pb de la selection de paramètres
# X. Add parameters
# X.X Add parameters.range
# X.X Add parameters.specific values
# X.X Ajouter plusieurs parametres differents
# 5. Tests 24/06
# 6. Docs 24/06
# 7. Gestion des mauvaise valeurs
# X. Add parallel
# 9. Gap
# 9.1 Implement PCA
# 9.2 Implement Gaussian
# 9.3 Essayer de rajouter les valeurs de gap et de safety

import warnings
import numpy as np

from abc import ABCMeta, abstractmethod
from functools import partial
from collections import defaultdict

from sklearn.base import clone, BaseEstimator
from sklearn.mixture.base import BaseMixture
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_array, check_random_state
from sklearn.utils.fixes import rankdata
from sklearn.utils.testing import ignore_warnings


class _NClusterSearchBase(six.with_metaclass(ABCMeta, object)):
    def __init__(self, estimator, parameters, n_jobs=1,
                 pre_dispatch='2*n_jobs', verbose=0):
        self.estimator = estimator
        self.parameters = parameters
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.verbose = verbose

    def _initialized(self, X, y):
        pass

    def _compute_values(self, n_clusters_list):
        return np.array(n_clusters_list)

    def _parameter_grid(self, X, y):
        # Determine the name of the property controling then number of clusters
        if isinstance(self.estimator, BaseMixture):
            # General case
            property_name = 'n_components'
        elif isinstance(self.estimator, BaseEstimator):
            # Mixture case
            property_name = 'n_clusters'
        else:
            raise ValueError("The estimator has to be from the type ...")

        # Compute n_clusters parameters
        parameters = self.parameters.copy()
        n_clusters_list = parameters.pop(property_name)
        self.n_clusters_values, self._index = np.unique(
            self._compute_values(n_clusters_list),
            return_index=True)
        self._index = self._index < len(n_clusters_list)

        for n_clusters in self.n_clusters_values:
            for params in ParameterGrid(parameters):
                params[property_name] = n_clusters
                yield params

    @abstractmethod
    def _estimator_fit(self, estimator, X, y, parameters):
        pass

    @abstractmethod
    def _compute_results(self, X, out):
        pass

    def fit(self, X, y):
        self._initialized(X, y)

        base_estimator = clone(self.estimator)

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )(delayed(self._estimator_fit)(clone(base_estimator), X, y, parameters)
          for parameters in self._parameter_grid(X, y))

        score, parameters = self._compute_results(X, out)
        score = np.array(score, dtype=np.float64).ravel()
        ranks = np.asarray(rankdata(-score, method='min'),
                           dtype=np.int32)

        self.results_ = dict()
        self.results_['score'] = score
        self.results_['rank_score'] = ranks
        # Use one np.MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(np.ma.masked_all,
                                            (len(parameters),), dtype=object))
        for cand_i, params in enumerate(parameters):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        self.results_.update(param_results)

        # Store a list of param dicts at the key 'params'
        self.results_['params'] = parameters

        # Ajouter support best_estimator une fois pandas géré
        # optimal_n_clusters = self.n_clusters_range[np.nanargmax(self.score_)]
        # self.best_estimator_ = self.estimator.set_params(
        #     n_clusters=optimal_n_clusters)
        self.best_estimator_ = self.estimator  # remove at the end

        return self


# Search for unsupervised metrics
class UnsupervisedMetricSearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters, metric):
        super(UnsupervisedMetricSearch, self).__init__(
            estimator, parameters)
        self.metric = metric

    def _estimator_fit(self, estimator, X, y, parameters):
        estimator.set_params(**parameters)
        try:
            score = self.metric(X, estimator.fit_predict(X))
        except ValueError:
            # TODO Add a warning for false values
            warnings.warn('Put a warning.')
            score = np.nan

        return score, parameters

    def _compute_results(self, X, out):
        scores, parameters = zip(*out)

        return np.array(scores), parameters


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

    def _initialized(self, X, y):
        n_samples, _ = X.shape
        rng = check_random_state(self.random_state)
        self.data_ = (
            rng.uniform(size=(self.n_draws, 2 * n_samples)) < self.p_samples)

    def _estimator_fit(self, estimator, X, y, parameters):
        estimator.set_params(**parameters)
        draw_scores = np.empty(self.n_draws)
        for l, d in enumerate(self.data_):
            p1, p2 = np.split(d, 2)

            labels1 = estimator.fit_predict(X[p1])
            labels2 = estimator.fit_predict(X[p2])

            draw_scores[l] = self.metric(labels1[p2[p1]], labels2[p1[p2]])
        score = draw_scores.mean()
        return score, parameters

    def _compute_results(self, X, out):
        scores, parameters = zip(*out)

        return np.array(scores), parameters


# Scorer for distortion jump
class DistortionJumpSearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters):
        super(DistortionJumpSearch, self).__init__(
            estimator, parameters)

    def _compute_values(self, n_clusters_list):
        return np.hstack((n_clusters_list, np.array(n_clusters_list) + 1))

    def _estimator_fit(self, estimator, X, y, parameters):
        _, n_features = X.shape
        estimator.set_params(**parameters)
        distortion = (np.array(estimator.fit(X).score(X)) /
                      n_features) ** (-n_features / 2)
        return distortion, parameters

    def _compute_results(self, X, out):
        distortion, parameters = zip(*out)
        distortion = np.array(distortion).reshape(
            len(self.n_clusters_values), -1)
        parameters = np.array(parameters).reshape(distortion.shape)
        return (distortion[self._index] - distortion[1:][self._index[:-1]],
                list(parameters[self._index].ravel()))


# Scorer for Pham
class PhamSearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters):
        super(PhamSearch, self).__init__(
            estimator, parameters)

    def _compute_values(self, n_clusters_list):
        return np.hstack((n_clusters_list, np.array(n_clusters_list) - 1))

    def _estimator_fit(self, estimator, X, y, parameters):
        estimator.set_params(**parameters)
        if parameters['n_clusters'] == 0:
            return np.nan, parameters
        return estimator.fit(X).score(X), parameters

    def _compute_results(self, X, out):
        # Regler le problèmes des paramètres qui ne sont pas les bons
        _, n_features = X.shape

        scores, parameters = zip(*out)

        scores = np.array(scores).reshape(len(self.n_clusters_values), -1)
        parameters = np.array(parameters).reshape(scores.shape)

        weights = 1. - np.exp((self.n_clusters_values[self._index] - 2) *
                              np.log(5. / 6.) + np.log(.75) -
                              np.log(n_features))

        with ignore_warnings(category=RuntimeWarning):
            scores = np.where(
                scores[self._index] != 0., scores[self._index] /
                (weights[:, np.newaxis] * scores[:-1][self._index[1:]]), 1.)

        if self.n_clusters_values[0] == 0:
            scores[0, :] = 1.

        return -scores, list(parameters[self._index].ravel())


# Scorer for Gap
class GapSearch(_NClusterSearchBase):
    def __init__(self, estimator, parameters, n_draws=10, n_jobs=1,
                 random_state=None):
        super(GapSearch, self).__init__(estimator, parameters, n_jobs)
        self.n_draws = n_draws
        self.random_state = random_state

    def _initialized(self, X, y):
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)

        # Compute the random data once for all
        bb_min, bb_max = np.min(X, 0), np.max(X, 0)
        self._data = (rng.uniform(size=(self.n_draws, n_samples, n_features)) *
                      (bb_max - bb_min) + bb_min)

    def _compute_values(self, n_clusters_list):
        return np.hstack((n_clusters_list, np.array(n_clusters_list) + 1))

    def _estimator_fit(self, estimator, X, y, parameters):
        estimator.set_params(**parameters)

        estimated_log_inertia = np.log(-estimator.fit(X).score(X))
        inertia_n_draws = np.empty(self.n_draws)
        for t, Xt in enumerate(self._data):
            inertia_n_draws[t] = -estimator.fit(Xt).score(Xt)
        inertia_n_draws = np.log(inertia_n_draws)

        expected_log_inertia = np.mean(inertia_n_draws)
        gap = expected_log_inertia - estimated_log_inertia
        safety = (np.std(inertia_n_draws) *
                  np.sqrt(1. + 1. / self.n_draws))

        return gap, safety, parameters

    def _compute_results(self, X, out):
        gap, safety, parameters = zip(*out)

        gap = np.array(gap).reshape(len(self.n_clusters_values), -1)
        safety = np.array(safety).reshape(gap.shape)
        parameters = np.array(parameters).reshape(gap.shape)

        scores = (gap[self._index] - gap[1:][self._index[:-1]] +
                  safety[1:][self._index[:-1]])

        return scores, list(parameters[self._index].ravel())


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

        self.results_ = self.scorer_.results_
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

    estimator = KMeans(random_state=0)

    n_clusters_range = np.arange(1, 8)
    parameters = {'n_clusters': np.arange(1, 8), 'random_state': [0, 0]}
    search = OptimalNClusterSearch(estimator=estimator, parameters=parameters,
                                   fitting_process='gap').fit(X)

    # print("Optimal number of clusters : %s" % search.best_estimator_)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])

    # print(search.results_)

    plt.figure()
    plt.plot(search.results_['param_n_clusters'],
             search.results_['score'], 'o-', alpha=.6)
    plt.show()
