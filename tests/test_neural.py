import numpy as np
from dexml.neural import *

p = 3
N = 100
M = 2
np.random.seed(2345)

X = np.random.rand(p, N) - 0.5
X[0] = np.ones(N)
y = X[1] + X[2] + np.random.normal(0, 0.2, N)


def test_slp_fitting_reduces_error():
    slp = SingleLayeredPerceptron(X, y, 2, gamma = 0.01)

    initial_error = np.sum((slp.predict(X) - y) ** 2)

    slp.fit(100)
    fitted_error = np.sum((slp.predict(X) - y) ** 2)

    assert fitted_error < initial_error
