"""Simple backpropagated neural nets."""


import numpy as np
import itertools


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

        self.n = X.shape[1]
        self.M = M
        self.K = 1
        self.p = p

        self.act_fn = act_fn
        self.act_fn_deriv = act_fn_deriv
        self.gamma = gamma

        self.alpha = self.initialize_w(n_neurons=M, n_inputs=p)
        self.beta = self.initialize_w(n_neurons=self.K, n_inputs=M)

    def update_weights(self):
        # Forward pass
        Zm = self.update_Zm(self.alpha, self.X)
        Tk = self.update_Tk(self.beta, Zm)
        fkX = Tk   # More activation here later?

        # Backwards pass
        dki = self.update_dki(fkX, Zm)
        smi = self.update_smi(dki, self.alpha, self.beta)

        # Gradients
        alpha_grad = self.get_alpha_gradient(smi, self.X)
        beta_grad = self.get_beta_gradient(dki, Zm)

        # Update weights
        self.beta = self.beta - (self.gamma * np.sum(beta_grad, axis=0))
        self.alpha = self.alpha - (self.gamma * np.sum(alpha_grad, axis=0))

    def update_Zm(self, alpha, X):
        Zm = []
        for alpha_m in alpha:
            aTX = np.dot(alpha_m, X)
            Zm.append(self.act_fn(aTX))

        return np.array(Zm)

    def update_Tk(self, beta, Zm):
        Tk = []
        for beta_k in beta:
            bTZ = np.dot(beta_k, Zm)
            Tk.append(bTZ)

        return np.array(Tk)

    def update_dki(self, fkX, Zm):
        dki = -2 * (self.y - fkX)
        return dki

    def update_smi(self, dki, alpha, beta):
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

    def predict(self, X):
        Zm = self.update_Zm(self.alpha, X)
        Tk = self.update_Tk(self.beta, Zm)
        return Tk[0]

    @staticmethod
    def initialize_w(n_neurons, n_inputs):
        return np.array([np.random.uniform(-0.1, 0.1, n_inputs)
                         for i in xrange(n_neurons)])
