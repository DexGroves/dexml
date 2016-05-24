import numpy as np


class CoordinateDescender(object):
    def __init__(self, loss, eps=1e-8, step=1e-4):
        self.loss = loss
        self.eps = eps
        self.step = step

    def descend(self, params):
        last_iter_loss = self.loss(params)

        while True:
            for p in xrange(len(params)):
                params = self.descend_one_dir(params, p, np.add)
                params = self.descend_one_dir(params, p, np.subtract)

            iter_loss = self.loss(params)
            if iter_loss / last_iter_loss <= (1 - self.eps):
                last_iter_loss = iter_loss
            else:
                break

        return params

    def descend_one_dir(self, params, p, fn):
        last_iter_loss = self.loss(params)
        while True:
            old_param = params[p]
            params[p] = fn(params[p], self.step)
            desc_loss = self.loss(params)

            if desc_loss < last_iter_loss:
                last_iter_loss = desc_loss
            else:
                params[p] = old_param
                break

        return params

