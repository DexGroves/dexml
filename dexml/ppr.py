"""Attempting to implement Projection Pursuit Regression from ESL."""


import numpy as np
from scipy.interpolate import UnivariateSpline
from dexml.ols import ols


class ProjectionPursuitRegressor(object):
    """Hold logic for PPR models."""
    def __init__(self, w, g):
        if len(g) == 0:
            self.g = [g]
        else:
            self.g = g

        self.w = np.atleast_2d(w)

    def predict(self, X):
        total = 0
        for i, w_i in enumerate(self.w):
            ridge_vec = self.g[i](np.dot(w_i, X.T))
            total += ridge_vec
        return total


def fit_ppr(X, y, fit_spline, M=1, w=None, g=None, eps=1e-8):
    """Fit a projection pursuit model."""
    if w is None:
        w = initialize_w(M, X.shape[1])

    g = [update_g(X, y, w_i, fit_spline) for w_i in w]

    last_error = np.inf
    iteration_error = ppr_sose(X, y, w, g)

    while (last_error - iteration_error > eps):
        last_error = iteration_error

        for i, _ in enumerate(w):
            other_w = [w_j for j, w_j in enumerate(w) if j != i][0]
            other_g = [g_j for j, g_j in enumerate(g) if j != i]
            y_residual = y - ppr_predict(X, other_w, other_g)
            w[i] = update_weights(X, y_residual, w[i], g[i])
            g[i] = update_g(X, y_residual, w[i], fit_spline)

        iteration_error = ppr_sose(X, y, w, g)

    return ProjectionPursuitRegressor(w, g)


def ppr_predict(X, w, g):
    w = np.atleast_2d(w)

    if len(g) == 0:
        g = [g]

    total = 0
    for i, w_i in enumerate(w):
        ridge_vec = g[i](np.dot(w_i, X.T))
        total += ridge_vec

    return total


def initialize_w(M, p):
    """Return the random starting weights."""
    return np.array([np.random.uniform(-1, 1, p) for i in xrange(M)])


def update_weights(X, y, w, g):
    w = np.atleast_2d(w)

    lhs = np.dot(w, X.T)
    rhs = (y - g(np.dot(w, X.T))) / g.derivative(1)(np.dot(w, X.T))
    target = lhs + rhs

    weights = g.derivative(1)(np.dot(w, X.T))**2

    w_new = ols(X, target[0], weights[0])

    return w_new


def update_g(X, y, w, fit_spline):
    return fit_spline(np.dot(w, X.T), y)


def fit_spline_generator(k, s):
    def fit_spline(x, y):
        return UnivariateSpline(x, y, k=k, s=s)

    return fit_spline


def ppr_sose(X, y, w, g):
    """Sum of squared error for a projection pursuit regressor."""
    yhat = ppr_predict(X, w, g)
    return np.sum((y - yhat)**2)
