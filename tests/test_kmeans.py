# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# from numpy.testing import assert_array_equal
# from dexml.kmeans import *


# N = 1500
# K = 2
# seed = 1234

# X, y = make_blobs(n_samples=N, random_state=seed)


# def test_kmeans_agrees_with_scikit():
#     sk_model = KMeans(n_clusters=2, random_state=seed)
#     sk_pred = sk_model.fit_predict(X)

#     dexml_model = kmeans(X, y, K, seed)
#     dexml_pred = dexml_model.predict(X)

#     assert_array_equal(sk_pred, dexml_pred)
