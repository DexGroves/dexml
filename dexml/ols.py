import numpy as np


def ols(X, y, w=None):
    """Weighted OLS with linear algebra. Big memory footprint."""
    if w is None:
        w = np.ones(len(y))

    XtWx = np.dot(X.T * w, X)
    XtWX_inv = np.linalg.inv(XtWx)
    XtWy = np.dot(X.T * w, y)

    B = np.dot(XtWX_inv, XtWy)

    return B
