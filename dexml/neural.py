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

    def __init__(self, X, y, p, M, act_fn, act_fn_deriv, gamma=0.01):
        self.X = X
        self.y = y

        self.act_fn = act_fn
        self.act_fn_deriv = act_fn_deriv
        self.gamma = gamma

        self.alpha = self.initialize_w(M, p)
        self.beta = self.initialize_w(M, 1)

    def update_weights(self):
        # Forward pass
        Zm = self.update_Zm(self.X)
        Tk = self.update_Tk(Zm)
        fkX = np.sum(Tk, axis=0)

        # Backwards pass
        dki = self.update_dki(fkX, Zm)
        smi = self.update_smi(dki)

        # Update weights
        beta_delta = np.dot(dki, Zm.T)
        alpha_delta = np.dot(smi, self.X)

        self.beta = self.beta - (self.gamma * beta_delta.T)
        self.alpha = self.alpha - (self.gamma * alpha_delta)

    def update_Zm(self, X):
        Zm = np.array([self.act_fn(np.dot(alpha_i, X.T))
                       for i, alpha_i in enumerate(self.alpha)])
        return Zm

    def update_Tk(self, Zm):
        Tk = np.array([Zm[i] * beta_i
                       for i, beta_i in enumerate(self.beta)])
        return Tk

    def update_dki(self, fkX, Zm):
        BTz = np.dot(self.beta.T, Zm)
        dki = -2 * (self.y - fkX) * BTz
        return dki

    def update_smi(self, dki):
        lhs = self.act_fn_deriv(np.dot(self.alpha, self.X.T))
        rhs = self.beta * dki
        return lhs * rhs

    @staticmethod
    def initialize_w(M, p):
        return np.array([np.random.uniform(-0.1, 0.1, p) for i in xrange(M)])
