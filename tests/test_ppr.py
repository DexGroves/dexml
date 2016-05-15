from dexml.ppr import *
import numpy as np


p = 5
n = 100
M = 1
np.random.seed(1234)

y = np.random.normal(0, 1, n)
X = np.random.rand(n, p) - 0.5  # No intercept

fit_spline = fit_spline_generator(2, None)


def test_ppr_reduces_error_for_m_eq_1():
    w_t = initialize_w(1, p)
    g_t = [update_g(X, y, w_t, fit_spline)]

    starting_error = ppr_sose(X, y, w_t, g_t)

    w_t = update_weights(X, y, w_t, g_t[0])
    g_t[0] = update_g(X, y, w_t, fit_spline)
    updated_error = ppr_sose(X, y, w_t, g_t)

    assert updated_error < starting_error
