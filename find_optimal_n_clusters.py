import numpy as np

from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import silhouette_score


def compute_silhouette_score(estimator, X, y=None):
    return (0. if estimator.n_clusters == 1 else
            silhouette_score(X, estimator.fit_predict(X)))


def compute_calinski_harabaz_score(estimator, X, y=None):
    return (0. if estimator.n_clusters == 1 else
            calinski_harabaz_score(X, estimator.fit_predict(X)))


def compute_stability_score(estimator, X, y=None, n_draws=10, p_samples=.8,
                            random_state=None):
    if estimator.n_clusters == 1:
        return 0.

    # Generate data
    rng = np.random.RandomState(random_state)
    n_samples, _ = X.shape
    data = rng.uniform(size=(n_draws, 2 * n_samples)) < p_samples

    score = np.empty(n_draws)
    for k, d in enumerate(data):
        p1, p2 = np.split(d, 2)

        labels1 = estimator.fit_predict(X[p1])
        labels2 = estimator.fit_predict(X[p2])

        score[k] = fowlkes_mallows_score(labels1[p2[p1]], labels2[p1[p2]])
    return score.mean()


def compute_distortion_jump_score(estimator, X, y=None):
    _, n_features = X.shape
    inertia_0 = (estimator.fit(X).score(X) / n_features) ** (-n_features / 2)
    next_estimator = estimator.set_params(n_clusters=estimator.n_clusters + 1)
    inertia_1 = (next_estimator.fit(X).score(X) /
                 n_features) ** (-n_features / 2)
    print(inertia_0, inertia_1, inertia_0 - inertia_1)
    return inertia_0 - inertia_1


def compute_pham_score(estimator, X, y=None):
    _, n_features = X.shape

    if estimator.n_clusters == 1:
        return -1.

    inertia_1 = estimator.fit(X).score(X)
    prev_estimator = estimator.set_params(n_clusters=estimator.n_clusters - 1)
    inertia_0 = prev_estimator.fit(X).score(X)

    if inertia_0 == 0.:
        return -1.

    weight = 1. - np.exp((estimator.n_clusters - 1) * np.log(5. / 6.) +
                         np.log(.75) - np.log(n_features))

    print(estimator.n_clusters + 1, inertia_1, weight, inertia_0)

    score = inertia_1 / (weight * inertia_0)

    return -score


class FindOptimalNClusters():
    """Compute the optimal number of cluster.

    Parameters
    ----------
    estimator : estimator object

    parameters :

    fitting_process : 'gap', 'distortion_jump', 'stability', 'silhouette', 'pham'

    verbose : int

    Attributes
    ----------
    best_estimator_ : estimator

    best_score_ : float

    best_params_ : dict
    """

    def __init__(self, estimator, n_clusters_max, fitting_process='gap',
                 verbose=0):
        self.estimator = estimator
        self.n_clusters_max = n_clusters_max
        self.fitting_process = fitting_process
        self.verbose = verbose

    def _compute_score(self, estimator, X, y=None):
        score_func = {
            'silhouette': compute_silhouette_score,
            'calinski_harabaz': compute_calinski_harabaz_score,
            'stability': compute_stability_score,
            'distortion_jump': compute_distortion_jump_score,
            'pham': compute_pham_score
        }
        return score_func[self.fitting_process](estimator, X, y)

    def fit(self, X, y=None):
        """Fit.

        Parameters
        ----------

        Returns
        -------

        """
        n_clusters_range = np.arange(1, self.n_clusters_max + 1)

        self.score_ = np.empty(len(n_clusters_range))
        for k, n_clusters in enumerate(n_clusters_range):
            # Set the n_cluster parameters
            self.estimator.set_params(n_clusters=n_clusters)

            # Compute the score for a specific n_cluster
            self.score_[k] = self._compute_score(self.estimator, X)

        # Final computation of the score
        optimal_n_clusters = n_clusters_range[np.argmax(self.score_)]
        self.best_estimator_ = self.estimator.set_params(
            n_clusters=optimal_n_clusters)

        return self


if __name__ == '__main__':
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs, make_circles, load_iris

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

    # estimator = GaussianMixture()
    search = FindOptimalNClusters(estimator=estimator, n_clusters_max=8,
                                  fitting_process='gap')
    search.fit(X)

    print("Optimal number of clusters : %s" % search.best_estimator_)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])

    print(search.score_)

    plt.figure()
    plt.plot(range(1, len(search.score_) + 1), search.score_, 'o-', alpha=.6)
    plt.show()
