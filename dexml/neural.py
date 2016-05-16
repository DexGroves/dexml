"""Simple backpropagated neural nets."""


import numpy as np


class SingleLayeredPerceptron(object):
    """Fully connected neural net with one hidden layer.
    Only does regression.
    """

    def __init__(self,
                 p,
                 M=None,
                 alpha=None,
                 beta=None,
                 activation=np.tanh):

        if alpha is None or beta is None:
            self.alpha = self.initialize_w(M, p)
            self.beta = self.initialize_w(M, 1)
            self.activation = activation

    def pass_forward(self, X):
        Zm = [self.activation(np.dot(alpha_i, X.T))
              for i, alpha_i in enumerate(self.alpha)]

        Tk = np.array([Zm[i] * beta_i for i, beta_i in enumerate(self.beta)])

        return np.sum(Tk, axis = 0)

    @staticmethod
    def initialize_w(M, p):
        """Return the random starting weights."""
        return np.array([np.random.uniform(-1, 1, p) for i in xrange(M)])
