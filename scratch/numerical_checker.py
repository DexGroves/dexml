from copy import copy
from scratch.scratch import *


p = 3
N = 500
M = 2
np.random.seed(2345)


X = np.random.rand(N, p) - 0.5
y = np.random.normal(0, 1, N)
y[X[:, 1] > 0] += 1
# y = logistic(np.random.normal(0, 1, N))
y = np.atleast_2d(y).T


nn = NeuralNetwork(p, [2], logistic, logistic_prime)
trainer = BFGSTrainer(nn)
trainer.fit(X, y)

nt = copy(trainer.mlp)
yhat = nt.prop_forward(X)
nt.prop_backward(X, y)

analytical_grad = nt.get_gradients()

old_w = nt.get_flat_weights()

test_errors = []
eps = 1e-9
initial_error = np.sum((y - yhat)**2)/2
for i, w in enumerate(old_w):
    test_w = copy(old_w)
    test_w[i] = w + eps
    nt.set_flat_weights(test_w)
    yhat_test = nt.prop_forward(X)
    test_error = np.sum((y - yhat_test)**2)/2

    test_errors.append(test_error)

numerical_grad = (test_errors - initial_error) / eps
analytical_grad / numerical_grad




