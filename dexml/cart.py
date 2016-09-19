"""CART: Classification and Regression Trees.
From ESL Chapter 9.
"""


import numpy as np


class TerminalNode(object):
    """Represent the end state of a node."""
    pass


class SplitNode(object):
    """Represent the split state of a node."""
    def __init__(self, j, s, c_left, c_right, g_left, g_right):
        self.j = j
        self.s = s
        self.c_left = c_left
        self.c_right = c_right
        self.g_left = g_left
        self.g_right = g_right


class CART(object):
    """Generic CART model, sans any distribution."""

    def __init__(self, X, y, max_depth):
        self.max_depth = max_depth
        self.tree = self.fit(X, y)

    def fit(self, X, y, depth=0):
        if depth == self.max_depth:
            return TerminalNode()

        best_split = self.find_best_split(X, y)

        if type(best_split) is TerminalNode:
            return TerminalNode()  # Forced, data split to one row
        else:
            j, s, c_left, c_right = best_split

        left_split = X[:, j] <= X[s, j]
        right_split = X[:, j] > X[s, j]

        X_left = X[left_split, ]
        X_right = X[right_split, ]

        y_left = y[left_split, ]
        y_right = y[right_split, ]

        g_left = self.fit(X_left, y_left, depth + 1)
        g_right = self.fit(X_right, y_right, depth + 1)

        return SplitNode(j, s, c_left, c_right, g_left, g_right)

    def predict(self, X, tree=None):
        yhat = np.zeros(X.shape[0])

        if tree is None:
            tree = self.tree

        if type(tree) is TerminalNode:
            return yhat

        left_split = X[:, tree.j] <= X[tree.s, tree.j]
        right_split = X[:, tree.j] > X[tree.s, tree.j]

        X_left = X[left_split, ]
        X_right = X[right_split, ]

        yhat[left_split] = tree.c_left + self.predict(X_left, tree.g_left)
        yhat[right_split] = tree.c_right + self.predict(X_right, tree.g_right)

        return yhat


class GaussCART(CART):
    """Gaussian-only CART model."""

    def find_best_split(self, X, y):
        """Scan over ALL X values to find the row, col split that
        minimises Gaussian error.
        """
        best_error = np.inf
        best_split = (0, 0)
        for j, xp in enumerate(X.T):
            if type(xp) is not np.ndarray:
                return TerminalNode()

            for s, xpi in enumerate(xp):
                left_split = xp <= xpi
                right_split = xp > xpi

                if sum(left_split) == 0 or sum(right_split) == 0:
                    continue

                c_left = self.fit_constant(y[left_split])
                c_right = self.fit_constant(y[right_split])

                err_js = np.sum((c_left - y[left_split])**2) + \
                    np.sum((c_right - y[right_split])**2)

                if err_js < best_error:
                    best_split = (j, s, c_left, c_right)

        return best_split

    @staticmethod
    def fit_constant(y):
        """Gaussian minimizer."""
        return np.mean(y)
