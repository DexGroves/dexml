"""Attempting to implement Projection Pursuit Regression from ESL."""


import numpy as np
from scipy.interpolate import UnivariateSpline
from dexml.ols import ols


def fit_spline_generator(k, s):
    def fit_spline(x, y):
        return UnivariateSpline(x, y, k=k, s=s)

    return fit_spline


def initialize_w(M, p):
    """Return the random starting weights."""
    w = np.array([np.random.uniform(-1, 1, p) for i in xrange(M)])
    return w


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


def ppr_sose(X, y, w, g):
    """Sum of squared error for a projection pursuit regressor."""
    yhat = ppr_predict(X, w, g)
    return np.sum((y - yhat)**2)


def ppr_predict(X, w, g):
    w = np.atleast_2d(w)

    total = 0
    for i, w_i in enumerate(w):
        ridge_vec = g[i](np.dot(w_i, X.T))
        total += ridge_vec

    return total
