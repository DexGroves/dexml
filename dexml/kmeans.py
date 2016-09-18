"""K-means clustering."""


import numpy as np
import random
from dexml.utils import euclidian_distance


class KMeans(object):
    """K-means cluster-er."""

    def __init__(self, X, y, K, seed, maxiter=300):
        random.seed(seed)

        self.X = X
        self.y = y
        self.K = K
        self.maxiter = maxiter

        self.dist = euclidian_distance

        self.C = self.initialize_centroids(self.X, self.K)
        self.fit()

    def fit(self):
        """Call fit_iterate until convergence."""
        for i in range(self.maxiter):
            self.fit_iterate()

    def fit_iterate(self):
        """Update cluster membership and cluster centers once."""
        self.closest_c = self.get_closest_c(self.X)
        self.assign_new_centroids()

    def get_closest_c(self, X):
        """Get the closest cluster centroids for each element of X."""
        closest_c = np.zeros(X.shape[0])
        for i, xi in enumerate(X):
            dists = np.array([self.dist(xi, ci) for ci in self.C])
            closest_c[i] = dists.argmin()
        return closest_c

    def assign_new_centroids(self):
        """Recalculate the cluster centroid based on its members."""
        for i, ci in enumerate(self.C):
            members = self.X[self.closest_c == i]
            new_mean = np.apply_along_axis(np.mean, 0, members)
            self.C[i, ] = new_mean

    def predict(self, X):
        return self.get_closest_c(X)

    @staticmethod
    def initialize_centroids(X, K):
        """Initialise centroids as random points from population.
        Possible that there is a better way. Also this
        assumes no duplicates!
        """
        cis = random.sample(range(X.shape[0]), K)
        return X[cis, ]
