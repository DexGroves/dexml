import numpy as np
from dexml.mlp import *
from dexml.utils import logistic, logistic_prime


np.random.seed(2345)

p = 3
N = 100
M = 5


X = np.random.rand(N, p) - 0.5
y = X[:, 0] + 0.5 * X[:, 1] + 2 * X[:, 2]
y = logistic(y)
y = np.atleast_2d(y).T


def test_slp_reduces_train_error():
    nn = NeuralNetwork(p, [M, M], logistic, logistic_prime)
    trainer = BFGSTrainer(nn)

    initial_error = np.sum((y - nn.prop_forward(X))**2) / 2
    trainer.fit(X, y)
    final_error = np.sum((y - nn.prop_forward(X))**2) / 2

    assert final_error < initial_error
    assert fit_result['success']
