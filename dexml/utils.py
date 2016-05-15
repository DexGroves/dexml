import numpy as np


def sose(y, yhat):
    return np.sum((y - yhat)**2)
