import numpy as np
from dexml.neural import *


def sigma(x):
    return 1.0 / (1 + np.exp(-x))


p = 2
N = 100
M = 2
np.random.seed(2345)

X = np.random.rand(N, p) - 0.5
y = sigma(X[:, 0] + 2 * X[:, 1])
y = np.atleast_2d(y).T


def test_slp_reduces_train_error():
    slp = SingleLayeredPerceptron(p, M)
    trainer = BFGSTrainer(slp)

    yh = trainer.slp.forward(X)
    initial_error = np.sum((yh - y)**2)

    fit_result = trainer.fit(X, y, 'BFGS')
    yh = trainer.slp.forward(X)
    fitted_error = np.sum((yh - y)**2)

    assert fitted_error < initial_error
    assert fit_result['success']
