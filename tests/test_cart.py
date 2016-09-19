import numpy as np
from dexml.cart import GaussCART
from dexml.utils import sose
from sklearn import tree


np.random.seed(2345)

p = 3
N = 100


X = np.random.rand(N, p) - 0.5
y = X[:, 0] + 0.5 * X[:, 1] + 2 * X[:, 2]


def test_cart_d1_agrees_with_scikit():
    d_cart = GaussCART(X, y, 1)
    d_pred = d_cart.predict(X)

    sk_cart = tree.DecisionTreeRegressor(max_depth=1)
    sk_cart = sk_cart.fit(X, y)
    sk_pred = sk_cart.predict(X)

    d_error = np.round(sose(y, d_pred), 6)
    sk_error = np.round(sose(y, sk_pred), 6)

    assert d_error == sk_error
