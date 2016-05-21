"""Simple backpropagated neural nets.
With help from https://github.com/stephencwelch/Neural-Networks-Demystified
"""


from scipy import optimize
import numpy as np


class Layer(object):
    """A fully connected layer of a neural network.
    Responsible for the input weights and representing its own activity."""
    def __init__(self, input_layer_size, M):
        self.W = self.initialise_weights(input_layer_size, M)

    def update_W(self, dW1):
        self.W = self.W + dW1

    def set_W(self, W):
        self.W = W

    @staticmethod
    def initialise_weights(n_lhs, n_rhs):
        return np.random.rand(n_lhs, n_rhs)


class SingleLayeredPerceptron(object):
    """Fully connected, one-hidden layer neural network."""
    def __init__(self, p, M):
        self.p = p
        self.M = M
        self.K = 1

        self.hidden = Layer(p, M)
        self.output = Layer(M, 1)

        self.sigma = lambda x: 1.0 / (1 + np.exp(-x))
        self.sigma_prime = lambda x: (np.exp(-1 * x) /
                                      ((1 + np.exp(-1 * x))**2))

    def forward(self, X):
        self.z2 = np.dot(X, self.hidden.W)
        self.a2 = self.sigma(self.z2)

        self.z3 = np.dot(self.a2, self.output.W)
        self.a3 = self.sigma(self.z3)

        return self.a3

    def backward(self, X, y):
        delta3 = np.multiply(-(y - self.a3), self.sigma_prime(self.z3))
        self.dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.output.W.T) * self.sigma_prime(self.z2)
        self.dJdW1 = np.dot(X.T, delta2)

        return self.dJdW1, self.dJdW2

    def get_weights(self):
        weights = np.concatenate((self.hidden.W.ravel(),
                                  self.output.W.ravel()))
        return weights

    def get_gradients(self, X, y):
        dJdW1, dJdW2 = self.backward(X, y)
        return np.concatenate((self.dJdW1.ravel(), self.dJdW2.ravel()))

    def set_weights(self, weights):
        alpha_start = 0
        alpha_end = self.M * self.p
        self.hidden.set_W(np.reshape(
            weights[alpha_start:alpha_end], (self.p, self.M)))
        beta_end = alpha_end + self.M * self.K
        self.output.set_W(np.reshape(
            weights[alpha_end:beta_end], (self.M, self.K)))


class BFGSTrainer(object):
    """Train an SLP with BFGS."""
    def __init__(self, slp):
        self.slp = slp

    def fit(self, X, y, method='BFGS'):
        weights0 = self.slp.get_weights()

        options = {'maxiter': 1000, 'disp': True}

        _res = optimize.minimize(self.cost_and_grad,
                                 weights0,
                                 jac=True,
                                 method=method,
                                 args=(X, y),
                                 options=options)

        self.slp.set_weights(_res.x)

        return _res

    def cost_function(self, X, y):
        yhat = self.slp.forward(X)
        return np.sum((y - yhat)**2)

    def cost_and_grad(self, weights, X, y):
        self.slp.set_weights(weights)

        cost = self.cost_function(X, y)
        grad = self.slp.get_gradients(X, y)

        return cost, grad
