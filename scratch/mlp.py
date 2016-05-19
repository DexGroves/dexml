import numpy as np


class Layer(object):
    """Stuff"""
    def __init__(self, input_layer_size, size, output_layer_size):
        self.W1 = self.initialise_weights(input_layer_size, size)
        self.W2 = self.initialise_weights(size, output_layer_size)

    def update_W1(self, dW1):
        self.W1 = self.W1 + dW1

    def update_W2(self, dW2):
        self.W2 = self.W2 + dW2

    @staticmethod
    def initialise_weights(n_lhs, n_rhs):
        return np.random.rand(n_lhs, n_rhs)


class SingleLayeredPerceptron(object):
    """Stuff"""
    def __init__(self, X, y, M=2, gamma=0.01):
        self.M = M
        self.n = X.shape[0]
        self.p = X.shape[1]

        self.X = X
        self.y = y

        self.hidden = Layer(self.p, self.M, 1)

        self.sigma = lambda x: 1.0/(1+np.exp(-x))
        self.sigma_prime = lambda x: (np.exp(-1*x) / ((1+np.exp(-1*x))**2))

    def forward(self, X):
        self.z2 = np.dot(X, self.hidden.W1)
        self.a2 = self.sigma(self.z2)
        self.z3 = np.dot(self.a2, self.hidden.W2)
        return self.sigma(self.z3)

    def backward(self, y):
        delta3 = np.multiply(-(y-self.z3), self.sigma_prime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.hidden.W2.T)*self.sigma_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def update_weights(self, dW1, dW2):
        self.hidden.update_W1(dw1 * self.gamma)
        self.hidden.update_W2(dw2 * self.gamma)


p = 3
N = 100
M = 2
np.random.seed(2345)

# X[0] = np.ones(N)
# y = X[1] + X[2] + np.random.normal(0, 0.2, N)

slp = SingleLayeredPerceptron(X, y, M)

X = np.random.rand(N, p) - 0.5
y = np.atleast_2d(slp.sigma(np.random.normal(0, 0.2, N))).T

yh = slp.forward(X)
dW1, dW2  = slp.backward(y)
np.sum((yh - y)**2)

slp.hidden.update_W1(0.001*dW1)
slp.hidden.update_W2(0.001*dW2)
yh = slp.forward(X)
dW1, dW2 = slp.backward(y)
np.sum((yh - y)**2)

