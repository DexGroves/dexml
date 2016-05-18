"""Simple backpropagated neural nets."""


import numpy as np
import itertools


class SingleLayeredPerceptron(object):
    """Fully connected neural net with one hidden layer.
    Only does regression.
    """

    def __init__(self,
                 X,
                 y,
                 M=2,
                 act_fn=None,
                 act_fn_deriv=None,
                 gamma=0.01):

        if act_fn is None or act_fn_deriv is None:
            act_fn=lambda x: 1.0/(1+np.exp(-x))
            act_fn_deriv=lambda x: (np.exp(-1*x) / ((1+np.exp(-1*x))**2))

        self.bp = BackPropagator(X,
                                 y,
                                 M,
                                 act_fn,
                                 act_fn_deriv,
                                 gamma)

    def fit(self, n_iter = 1000):
        for i in xrange(n_iter):
            self.bp.update_weights()

    def predict(self, X):
        Zm = self.bp.get_Zm(self.bp.alpha, X)
        Tk = self.bp.get_Tk(self.bp.beta, Zm)
        return Tk[0]


class BackPropagator(object):
    """Handle updating the weights of an SLP."""

    def __init__(self, X, y, M, act_fn, act_fn_deriv, gamma=0.01):
        self.X = X
        self.y = y

        self.n = X.shape[1]
        self.p = X.shape[0]
        self.M = M
        self.K = 1

        self.act_fn = act_fn
        self.act_fn_deriv = act_fn_deriv
        self.gamma = gamma

        self.alpha = self.initialize_w(n_neurons=M, n_inputs=self.p)
        self.beta = self.initialize_w(n_neurons=self.K, n_inputs=M)

    def update_weights(self):
        # Forward pass
        Zm = self.get_Zm(self.alpha, self.X)
        Tk = self.get_Tk(self.beta, Zm)
        fkX = Tk   # More activation here later?

        self.error = (self.y - fkX)**2

        # Backwards pass
        dki = self.get_dki(fkX, Zm)
        smi = self.get_smi(dki, self.alpha, self.beta)

        # Gradients
        alpha_grad = self.get_alpha_gradient(smi, self.X)
        beta_grad = self.get_beta_gradient(dki, Zm)

        # Update weights
        self.beta = self.beta - (self.gamma * np.sum(beta_grad, axis=0))
        self.alpha = self.alpha - (self.gamma * np.sum(alpha_grad, axis=0))

    def get_Zm(self, alpha, X):
        Zm = []
        for alpha_m in alpha:
            aTX = np.dot(alpha_m, X)
            Zm.append(self.act_fn(aTX))

        return np.array(Zm)

    def get_Tk(self, beta, Zm):
        Tk = []
        for beta_k in beta:
            bTZ = np.dot(beta_k, Zm)
            Tk.append(bTZ)

        return np.array(Tk)

    def get_dki(self, fkX, Zm):
        dki = -2 * (self.y - fkX)
        return dki

    def get_smi(self, dki, alpha, beta):
        smi = []
        for i in xrange(self.n):
            sm = []
            for m, alpha_m in enumerate(alpha):
                aTx = np.dot(alpha_m, self.X[:,i])
                lhs = self.act_fn_deriv(aTx)

                rhs = []
                for k, beta_k in enumerate(beta):
                    bTd = (beta_k[m] * dki[k][i])
                    rhs.append(bTd)
                rhs = np.sum(rhs)

                sm.append(lhs * rhs)
            smi.append(np.array(sm))

        return np.array(smi)

    def get_alpha_gradient(self, smi, X):
        alpha_grad = np.zeros((self.n, self.M, self.p))
        for i, m, l in itertools.product(
            xrange(self.n), xrange(self.M), xrange(1, self.p)):
            alpha_grad[i, m, l] = smi[i][m] * X[l, i]
        return alpha_grad

    def get_beta_gradient(self, dki, Zm):
        beta_grad = np.zeros((self.n, self.K, self.M))
        for i, k, m in itertools.product(
            xrange(self.n), xrange(self.K), xrange(1, self.M)):
            beta_grad[i, k, m] = dki[k][i] * Zm[m][i]
        return beta_grad

    @staticmethod
    def initialize_w(n_neurons, n_inputs):
        return np.array([np.random.uniform(-0.1, 0.1, n_inputs)
                         for i in xrange(n_neurons)])
