"""Simple backpropagated neural nets."""


import numpy as np


class SingleLayeredPerceptron(object):
    """Fully connected neural net with one hidden layer.
    Only does regression.
    """

    def __init__(self,
                 p,
                 M=None,
                 act_fn=np.tanh,
                 act_fn_deriv=lambda z: 1/np.cosh(z)):

        self.back_propagator = BackPropagator(p, M, act_fn, act_fn_deriv)


class BackPropagator(object):
    """Handle updating the weights of an SLP."""

    def __init__(self, X, y, p, M, act_fn, act_fn_deriv):
        self.X = X
        self.y = y

        self.act_fn = act_fn
        self.act_fn_deriv = act_fn_deriv

        self.alpha = self.initialize_w(M, p)
        self.beta = self.initialize_w(M, 1)

    def update_weights(self):
        fkX = pass_forward(self)


    def pass_forward(self):
        """Perform one forward pass through the network to obtain yhat.
        Stores intermediate Zk and Tk to avoid recomputation later.
        """
        self.Zm = self.get_Zm()
        self.Tk = self.get_Tk()

        return np.sum(self.Tk, axis = 0)

    def pass_backwards(self, fkX):
        """Perform one backward pass through network."""
        self.dki = self.get_dki(fkX)
        self.smi = self.get_smi()

    def get_Zm(self):
        return np.array([self.act_fn(np.dot(alpha_i, self.X.T))
                         for i, alpha_i in enumerate(self.alpha)])

    def get_Tk(self):
        return np.array([self.Zm[i] * beta_i
                         for i, beta_i in enumerate(self.beta)])

    def get_dki(self, fkX):
        BTz = np.dot(self.beta.T, self.Zm)
        return -2 * (self.y - fkX) * BTz

    def get_smi(self):
        lhs = self.act_fn_deriv(np.dot(self.alpha, self.X.T))
        rhs = self.beta * self.dki
        return lhs * rhs

    @staticmethod
    def initialize_w(M, p):
        """Return random starting weights."""
        return np.array([np.random.uniform(-1, 1, p) for i in xrange(M)])
