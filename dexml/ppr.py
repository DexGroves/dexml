"""Attempting to implement Projection Pursuit Regression from ESL."""


import numpy as np
from scipy.interpolate import UnivariateSpline
from dexml.ols import ols


def fit_spline_generator(k, s):
    def fit_spline(x, y):
        return UnivariateSpline(x, y, k=k, s=s)

    return fit_spline


def update_weights(X, y, w, g):
    lhs = np.dot(w, X.T)
    rhs = (y - g(np.dot(w, X.T))) / g.derivative(1)(np.dot(w, X.T))
    target = lhs + rhs

    weights = g.derivative(1)(np.dot(w, X.T))**2

    w_new = ols(X, target[0], weights[0])

    return np.atleast_2d(w_new)


def update_g(X, y, w, fit_spline):
    return fit_spline(np.dot(w, X.T), y)


def initialize_w(M, p):
    """Return the random starting weights."""
    w = np.array([np.random.uniform(-1, 1, p) for i in xrange(M)])
    return w


def ppr_sose(X, y, w, g):
    """Sum of squared error for a projection pursuit regressor."""
    yhat_total = np.zeros(len(y))
    for wm in w:
        ridge_vec = np.dot(wm, X)
        ridge_fn_output = g(y, ridge_vec)
        yhat_total += ridge_fn_output

    return np.sum((y - yhat_total)**2)
