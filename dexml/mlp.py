"""Simple backpropagated neural nets.
With help from https://github.com/stephencwelch/Neural-Networks-Demystified
"""


from scipy import optimize
from dexml.utils import logistic, logistic_prime
import numpy as np


class Layer(object):
    def __init__(self, P, M, sigma):
        self.P = P
        self.M = M
        self.n_weights = P * M
        self.W = self.initialise_weights(P, M)
        self.sigma = sigma

    def forward(self, X):
        return np.dot(X, self.W)

    def get_weights(self):
        return self.W.ravel()

    def set_weights(self, Wflat):
        self.W = np.reshape(Wflat, (self.P, self.M))

    @staticmethod
    def initialise_weights(P, M):
        return np.random.rand(P, M)


class NeuralNetwork(object):
    def __init__(self, P, Ms, sigma, sigma_prime):
        self.P = P
        self.K = 1
        self.Ms = Ms

        self.depth = len(Ms)

        self.sigma = sigma
        self.sigma_prime = sigma_prime

        self.layers = self.instantiate_layers()

    def prop_forward(self, X):
        Z1 = np.dot(X, self.layers[0].W)
        A1 = self.sigma(Z1)

        self.Zs = [Z1]
        self.As = [A1]

        for i, layer in enumerate(self.layers[1:]):
            Zn = np.dot(self.As[i], layer.W)
            An = self.sigma(Zn)

            self.Zs.append(Zn)
            self.As.append(An)

        return An

    def prop_backward(self, X, y):
        layers_rev = list(reversed(self.layers))
        Zs_rev = list(reversed(self.Zs))
        As_rev = list(reversed(self.As))
        As_rev.append(X)

        delta0 = np.multiply(-(y - As_rev[0]), self.sigma_prime(Zs_rev[0]))
        djdw0 = np.dot(As_rev[1].T, delta0)

        self.deltas = [delta0]
        self.djdws = [djdw0]

        for i in xrange(0, len(layers_rev) - 1):
            delta_n = np.dot(self.deltas[i], layers_rev[i].W.T) * \
                self.sigma_prime(Zs_rev[i + 1])
            djdw_n = np.dot(As_rev[i + 2].T, delta_n)

            self.deltas.append(delta_n)
            self.djdws.append(djdw_n)

        self.deltas = list(reversed(self.deltas))
        self.djdws = list(reversed(self.djdws))

    def instantiate_layers(self):
        layers = [Layer(self.P, self.Ms[0], self.sigma)]
        final_layer = Layer(self.Ms[self.depth - 1], self.K, self.sigma)

        for i in xrange(len(self.Ms) - 1):
            layers.append(Layer(self.Ms[i], self.Ms[i + 1], self.sigma))

        layers.append(final_layer)
        return layers

    def get_flat_weights(self):
        flat_weights = [layer.get_weights() for layer in self.layers]
        return np.concatenate(flat_weights)

    def get_gradients(self):
        flat_grads = [djdw.ravel() for djdw in self.djdws]
        return np.concatenate(flat_grads)

    def set_flat_weights(self, weights):
        iw = 0
        for i, layer in enumerate(self.layers):
            self.layers[i].set_weights(weights[iw:(iw + layer.n_weights)])
            iw += layer.n_weights


class BFGSTrainer(object):
    """Train an MLP with BFGS."""
    def __init__(self, mlp):
        self.mlp = mlp

    def fit(self, X, y, method='BFGS'):
        weights0 = self.mlp.get_flat_weights()

        options = {'maxiter': 1000, 'disp': True}

        _res = optimize.minimize(self.cost_and_grad,
                                 weights0,
                                 jac=True,
                                 method=method,
                                 args=(X, y),
                                 options=options)

        self.mlp.set_flat_weights(_res.x)

        return _res

    def cost_and_grad(self, weights, X, y):
        self.mlp.set_flat_weights(weights)

        yhat = self.mlp.prop_forward(X)
        self.mlp.prop_backward(X, y)

        cost = self.cost_function(y, yhat)
        grad = self.mlp.get_gradients()

        return cost, grad

    @staticmethod
    def cost_function(y, yhat):
        return np.sum((y - yhat)**2) / 2
