"""Attempting to implement Projection Pursuit Regression from ESL."""


import numpy as np
from scipy.interpolate import UnivariateSpline


def get_spline_generator(k, s):
    """Return a UnivariateSpline function for fixed k and s.

    Returned function will accept x, y and evaluate to the fitted
    values and the fitted derivative as numpy arrays.
    """

    def fit_spline(x, y):
        yhat = UnivariateSpline(x, y, k=k, s=s)(x)
        dyhat_dx = UnivariateSpline(x, y, k=k, s=s).derivative(1)(x)
        return yhat, dyhat_dx

    return fit_spline


def initialize_w(M, p):
    """Return the random starting weights."""
    w = [np.random.uniform(-1, 1, p) for i in xrange(M)]
    return w


def ppr_sose(X, y, w, g):
    """Sum of squared error for a projection pursuit regressor."""
    yhat_total = np.zeros(len(y))
    for wm in w:
        ridge_vec = np.dot(wm, X)
        ridge_fn_output = g(y, ridge_vec)[0]
        yhat_total += ridge_fn_output

    return np.sum((y - yhat_total)**2)
