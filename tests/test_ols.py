from dexml.ols import ols
import numpy as np
from sklearn import linear_model

p = 5
n = 100

np.random.seed(1234)

y = np.random.normal(0, 1, n)

covariates = np.random.rand(n, p) - 0.5
intercept = np.ones(n)
X = np.c_[intercept, covariates]


def test_ols_agrees_with_numpy():
    dexml_B = ols(X, y)
    numpy_B, _, _, _ = np.linalg.lstsq(X, y)

    assert np.allclose(dexml_B, numpy_B)


def test_weighted_ols_agrees_with_scikit():
    w = np.random.uniform(0, 1, n)

    dexml_B = ols(X, y, w)

    clf = linear_model.LinearRegression(fit_intercept=False)
    scikit_B = clf.fit(X, y, w).coef_

    assert np.allclose(dexml_B, scikit_B)
