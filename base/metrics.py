import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.metrics import mean_absolute_error, log_loss, balanced_accuracy_score, mean_squared_log_error
from pmdarima.metrics import smape

def optuna_metric(y_true, y_pred) -> float:
    # return smape(a=y_true, f=y_pred)
    # TODO: try msle but fix issue with negative values.
    # return mean_squared_log_error(y_true=y_true, y_pred=y_pred)
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)


def classification_metric(y_true, y_pred) -> float:
    return balanced_accuracy_score(y_true, y_pred)


def cv_metric(estimator, X, y) -> float:
    y_hat = estimator.predict(X)
    return smape(y, y_hat)


def calculate_smape(actual, predicted) -> float:

    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)

    return round(np.mean(np.abs(predicted - actual) / ((np.abs(predicted) + np.abs(actual)) / 2)) * 100, 2)


# def smape(a, f):
#     """
#     a --> actual (y_true)
#     f --> forecast (y_pred)
#     """
#     return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)


def mape(A, F):
    tmp = np.abs(A - F) / np.abs(A)
    len_ = len(tmp)
    return 100 * np.sum(tmp) / len_


def adjusted_r_squared(X, Y, r2):
    '''
    Returns a computed Adjusted R-Squared Coefficient.

    Parameters
    ----------
    X : Pandas DataFrame
        A pandas DataFrame including all the independant variables. Could be a series if there is only one predictor.

    Y : Pandas DataFrame or Series
        Labels or response variables 'Y'.

    r2 : float
        R-Squared Coefficient
    '''
    n = len(Y)
    p = X.shape[1]
    adj_r = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

    return adj_r


# https://stackoverflow.com/a/60666838
# compute the diameter based on convex hull
def _diameter(pts):
    # need at least 3 points to construct the convex hull
    if pts.shape[0] <= 1:
        return 0
    if pts.shape[0] == 2:
        return ((pts[0] - pts[1])**2).sum()
    # two points which are fruthest apart will occur as vertices of the convex hull
    hull = spatial.ConvexHull(pts)
    #   candidates = pts[spatial.ConvexHull(pts).vertices]
    candidates = pts[hull.vertices]
    return spatial.distance_matrix(candidates, candidates).max()


def dunn_index(pts, labels):
    # O(k n log(n)) with k clusters and n points; better performance with more even clusters
    max_intracluster_dist = pd.DataFrame(pts).groupby(labels).agg(_diameter)[0].max()
    centroids = pd.DataFrame(pts).groupby(labels).mean()
    # O(k^2) with k clusters; can be reduced to O(k log(k))
    # get pairwise distances between centroids
    cluster_dmat = spatial.distance_matrix(centroids, centroids)
    # fill diagonal with +inf: ignore zero distance to self in "min" computation
    cluster_dmat_mod = cluster_dmat + (np.eye(cluster_dmat.shape[0]) * np.inf)
    min_intercluster_dist = cluster_dmat_mod.min()
    return min_intercluster_dist / max_intracluster_dist
