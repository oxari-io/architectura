import numpy as np
import pandas as pd
from pmdarima.metrics import smape
from pmdarima.utils import check_endog
from scipy import spatial
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error


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


def mape(A, F):
    tmp = np.abs(A - F) / np.abs(A)
    len_ = len(tmp)
    return 100 * np.sum(tmp) / len_


# TODO: Use (MdAPE & sMdAPE): https://support.numxl.com/hc/en-us/articles/115001223503-MdAPE-Median-Absolute-Percentage-Error
def mdape(y_true, y_pred):
    r"""Compute the Median Absolute Percentage Error.

    sMdAPE is less intuitive, for example an MdAPE of 8% does not mean that the average absolute percentage error is 8%. 
    Instead it means that half of the absolute percentage errors are less than 8% and half are over 8%. 
    Defined as follows:

        :math:`\mathrm{sMdAPE} = \mathrm{median}(s_1,s_2,\cdots,s_N)}`

    Parameters
    ----------
    y_true : array-like, shape=(n_samples,)
        The true test values of y.

    y_pred : array-like, shape=(n_samples,)
        The forecasted values of y.


    References
    ----------
    .. [1] https://support.numxl.com/hc/en-us/articles/115001223503-MdAPE-Median-Absolute-Percentage-Error
    """    # noqa: E501
    y_true = check_endog(
        y_true,
        copy=False,
        preserve_series=False,
    )
    y_pred = check_endog(
        y_pred,
        copy=False,
        preserve_series=False,
    )
    abs_diff = np.abs(y_pred - y_true)
    return np.median((abs_diff * 100 / np.abs(y_true)))


def smdape(y_true, y_pred):
    r"""Compute the Symmetric Median Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like, shape=(n_samples,)
        The true test values of y.

    y_pred : array-like, shape=(n_samples,)
        The forecasted values of y.


    References
    ----------
    .. [1] https://support.numxl.com/hc/en-us/articles/115001223503-MdAPE-Median-Absolute-Percentage-Error
    """    # noqa: E501
    y_true = check_endog(
        y_true,
        copy=False,
        preserve_series=False,
    )
    y_pred = check_endog(
        y_pred,
        copy=False,
        preserve_series=False,
    )
    abs_diff = np.abs(y_pred - y_true)
    return np.median((abs_diff * 200 / (np.abs(y_pred) + np.abs(y_true))))


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
