"""General reusable data bits."""


import numpy as np


def sose(y, yhat):
    return np.sum((y - yhat)**2)


def logistic(z):
    return 1.0 / (1 + np.exp(-z))


def logistic_prime(z):
    return (np.exp(-1 * z) / ((1 + np.exp(-1 * z))**2))
