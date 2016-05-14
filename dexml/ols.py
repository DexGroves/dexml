import numpy as np


def ols(X, y):
    """OLS Regression with linear algebra. Big memory footprint."""
    XtX_inv = np.linalg.inv(np.dot(X.T, X))
    Xty = np.dot(X.T, y)

    B = np.dot(XtX_inv, Xty)

    return B
