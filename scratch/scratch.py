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
        return self.W.ravel

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
        djdw0 = np.dot(As_rev[1], delta0)

        self.deltas = [delta0]
        self.djdws = [djdw0]

        for i in xrange(0, len(layers_rev) - 1):
            delta_n = np.dot(delta0, layers_rev[i].W) * \
                self.sigma_prime(Zs_rev[i + 1])
            djdw_n = np.dot(As_rev[i + 2], delta_n)

            self.deltas.append(delta_n)
            self.djdws.append(djdw_n)

    def instantiate_layers(self):
        layers = [Layer(self.P, self.Ms[0], self.sigma)]

        for i in xrange(len(self.Ms) - 1):
            layers.append(Layer(self.Ms[i], self.Ms[i + 1], self.sigma))

        layers.append(Layer(self.Ms[i + 1], self.K, self.sigma))
        return layers

    def get_flat_weights(self):
        flat_weights = [layer.get_weights() for layer in self.layers]
        return np.concatenate(flat_weights)

    def set_flat_weights(self, weights):
        i = 0
        for i, layer in enumerate(self.layers):
            self.layers[i].set_weights(weights[i:layer.n_weights])
            i += layer.n_weights

    def get_gradients(self):
        flat_grads = [djdw.ravel() for djdw in self.djdws]
        return np.concatenate(flat_grads)



p = 2
N = 100
M = 2
np.random.seed(2345)

X = np.random.rand(N, p) - 0.5
# y set_weights sigma(np.atleast_2d(X[:, 1] + 2 * X[:, 2])).T
# + np.random.normal(0, 0.2, N)).T
# y = np.zeros(N)
# y[X[:, 1] > 0] get_gradients(self):
# = 1
y = logistic(np.random.normal(0, 1, N))
y = np.atleast_2d(y).T

nn = NeuralNetwork(p, [3, 3], logistic, logistic_prime)
nn.prop_forward(X)
nn.prop_backward(X, y)

# slp = SingleLayeredPerceptron(p, M)

# yh = slp.forward(X)
# np.sum((yh - y)**2)

# trainer = BFGSTrainer(slp)
# trainer.fit(X, y)

# yh = slp.forward(X)
# np.sum((yh - y)**2)

# # trainer.cost_and_grad(trainer.slp.get_weights(), X, y)


# # slp = SingleLayeredPerceptron(p, M)
# # trainer = BFGSTrainer(slp)
# # w = trainer.slp.get_weights()
# # w
# # np.sum((slp.forward(X) - y)**2)
# # trainer.slp.set_weights(w - 0.01)
# # w = trainer.slp.get_weights()
# # w
# # np.sum((slp.forward(X) - y)**2)
