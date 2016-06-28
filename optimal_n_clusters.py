import warnings
import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn.externals import six
from sklearn.utils import check_array, check_random_state
from sklearn.utils.testing import ignore_warnings


class _NClusterSearchBase(six.with_metaclass(ABCMeta, object)):
    def __init__(self, estimator, n_clusters_range):
        self.estimator = estimator
        self.n_clusters_range = n_clusters_range

    @abstractmethod
    def _compute_score(self, X, y=None):
        pass

    def fit(self, X, y=None):
        self.score_ = self._compute_score(X, y)

        optimal_n_clusters = self.n_clusters_range[np.nanargmax(self.score_)]
        self.best_estimator_ = self.estimator.set_params(
            n_clusters=optimal_n_clusters)

        return self


# Search for unsupervised metrics
class UnsupervisedMetricSearch(_NClusterSearchBase):
    def __init__(self, estimator, n_clusters_range, metric):
        super(UnsupervisedMetricSearch, self).__init__(
            estimator, n_clusters_range)
        self.metric = metric

    def _compute_score(self, X, y=None):
        score = np.empty(len(self.n_clusters_range))

        for k, n_clusters in enumerate(self.n_clusters_range):
            try:
                self.estimator.set_params(n_clusters=n_clusters)
                score[k] = self.metric(X, self.estimator.fit_predict(X))
            except ValueError:
                # TODO Add a warning for false values
                warnings.warn('Put a warning.')
                score[k] = np.nan

        return score


# Scorer for supervised metrics
class StabilitySearch(_NClusterSearchBase):
    def __init__(self, estimator, n_clusters_range, metric, n_draws=10,
                 p_samples=.8, random_state=None):
        super(StabilitySearch, self).__init__(
            estimator, n_clusters_range)
        self.metric = metric
        self.n_draws = n_draws
        self.p_samples = p_samples
        self.random_state = random_state

    def _compute_score(self, X, y=None):
        rng = check_random_state(self.random_state)

        # Compute the random data once for all
        n_samples, _ = X.shape
        self.data_ = (
            rng.uniform(size=(self.n_draws, 2 * n_samples)) < self.p_samples)

        score = np.empty(len(self.n_clusters_range))
        for k, n_clusters in enumerate(self.n_clusters_range):
            self.estimator.set_params(n_clusters=n_clusters)
            draw_scores = np.empty(self.n_draws)
            for l, d in enumerate(self.data_):
                p1, p2 = np.split(d, 2)

                labels1 = self.estimator.fit_predict(X[p1])
                labels2 = self.estimator.fit_predict(X[p2])

                draw_scores[l] = self.metric(labels1[p2[p1]], labels2[p1[p2]])
            score[k] = draw_scores.mean()

        return score


# Scorer for distortion jump
class DistortionJumpSearch(_NClusterSearchBase):
    def __init__(self, estimator, n_clusters_range):
        super(DistortionJumpSearch, self).__init__(
            estimator, n_clusters_range)

    def _compute_score(self, X, y=None):
        _, n_features = X.shape

        distortion_range = np.arange(self.n_clusters_range.min(),
                                     self.n_clusters_range.max() + 2)
        distortion = np.empty(len(distortion_range))
        for k, n_clusters in enumerate(distortion_range):
            self.estimator.set_params(n_clusters=n_clusters)
            distortion[k] = ((self.estimator.fit(X).score(X) / n_features) **
                             (-n_features / 2))
        return distortion[:-1] - distortion[1:]


# Scorer for Pham
class PhamSearch(_NClusterSearchBase):
    def __init__(self, estimator, n_clusters_range):
        super(PhamSearch, self).__init__(
            estimator, n_clusters_range)

    def _compute_score(self, X, y=None):
        _, n_features = X.shape

        score = np.empty(len(self.n_clusters_range))
        for k, n_clusters in enumerate(self.n_clusters_range):
            self.estimator.set_params(n_clusters=n_clusters)
            score[k] = self.estimator.fit(X).score(X)

        weights = 1. - np.exp((self.n_clusters_range[:-1] - 1) *
                              np.log(5. / 6.) + np.log(.75) -
                              np.log(n_features))

        with ignore_warnings(category=RuntimeWarning):
            score[1:] = np.where(score[:-1] != 0.,
                                 score[1:] / (weights * score[:-1]), 1.)
        score[0] = 1.

        return -score


# Scorer for Gap
class GapSearch(_NClusterSearchBase):
    def __init__(self, estimator, n_clusters_range, n_draws=10,
                 random_state=None):
        super(GapSearch, self).__init__(estimator, n_clusters_range)
        self.n_draws = n_draws
        self.random_state = random_state

    def _compute_score(self, X, y=None):
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)

        # Compute the random data once for all
        bb_min, bb_max = np.min(X, 0), np.max(X, 0)
        data = (rng.uniform(size=(self.n_draws, n_samples, n_features)) *
                (bb_max - bb_min) + bb_min)

        # Compute inertia for X and rand data
        inertia_range = np.arange(self.n_clusters_range.min(),
                                  self.n_clusters_range.max() + 2)
        inertia = np.empty(len(inertia_range))
        rand_inertia = np.empty((len(inertia_range), self.n_draws))

        for k, n_clusters in enumerate(inertia_range):
            self.estimator.set_params(n_clusters=n_clusters)

            inertia[k] = -estimator.fit(X).score(X)

            for l, Xt in enumerate(data):
                rand_inertia[k, l] = -estimator.fit(Xt).score(Xt)

        # Compute the gap score
        gap = np.mean(np.log(rand_inertia), 1) - np.log(inertia)
        safety = (np.std(np.log(rand_inertia), axis=1) *
                  np.sqrt(1. + 1. / self.n_draws))

        return gap[:-1] - gap[1:] + safety[1:]


# Ajouter les arguments qui remplacent
class OptimalNClusterSearch():
    def __init__(self, estimator, n_clusters_range, fitting_process='auto',
                 **kwargs):
        self.estimator = estimator
        self.n_clusters_range = n_clusters_range
        self.fitting_process = fitting_process
        self.kwargs = kwargs

    def _check_initial_parameters(self, X):
        n_samples, _ = X.shape

        X = check_array(X, dtype=[np.float64, np.float32])

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
            estimator=self.estimator, n_clusters_range=self.n_clusters_range,
            **self.kwargs)

    def fit(self, X, y=None):
        self._check_initial_parameters(X)

        self._select_fitting_method()
        self.scorer_.fit(X, y)

        self.score_ = self.scorer_.score_
        self.best_estimator_ = self.scorer_.best_estimator_
        self.n_clusters_range = self.scorer_.n_clusters_range

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

    # iris = load_iris()
    # X = iris.data

    estimator = KMeans()

    n_clusters_range = np.arange(1, 8)
%timeit search = OptimalNClusterSearch(estimator=estimator, n_clusters_range=n_clusters_range, fitting_process='gap', random_state=0).fit(X)

    # search = OptimalNClusterSearch(estimator=estimator,
    #                                n_clusters_range=n_clusters_range,
    #                                fitting_process='gap').fit(X)

    # print("Optimal number of clusters : %s" % search.best_estimator_)

    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1])

    # plt.figure()
    # plt.plot(n_clusters_range, search.score_, 'o-', alpha=.6)
    # plt.show()