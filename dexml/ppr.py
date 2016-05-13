"""Attempting to implement Projection Pursuit Regression from ESL."""


import numpy as np
from scipy.interpolate import UnivariateSpline


def get_spline_generator(k, s):
    """Return a UnivariateSpline function for fixed k and s.

    Returned function will accept x, y and evaluate to the fitted
    values and the fitted derivative as numpy arrays.
    """

    def fit_spline(x, y):
        yhat = UnivariateSpline(x, y, k = k, s = s)(x)
        dyhat_dx = UnivariateSpline(x, y, k = k, s = s).derivative(1)(x)
        return yhat, dyhat_dx

    return fit_spline
