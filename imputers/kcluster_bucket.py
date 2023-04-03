from typing import Union
from typing_extensions import Self
import kmedoids
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.impute import SimpleImputer, KNNImputer

from base.common import DefaultClusterEvaluator, OxariImputer
from base.helper import replace_ft_num
from base.mappings import NumMapping
from base.oxari_types import ArrayLike

from .core import BucketImputerBase


class KMeansBucketImputer(BucketImputerBase):
    def __init__(self, buckets_number=3, **kwargs):
        super().__init__(**kwargs)
        self.bucket_number = buckets_number
        self._estimator = KNNImputer(n_neighbors=self.bucket_number)


    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """
        self._estimator = self._estimator.fit(X.filter(regex="^ft_num"))
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        X_num = X.filter(regex="^ft_num")
        X_new = self._estimator.transform(X_num)
        X_new = pd.DataFrame(X_new, X.index, X_num.columns)
        return replace_ft_num(X, X_new)

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)


class KMedianBucketImputer(BucketImputerBase):
    def __init__(self, buckets_number=3, **kwargs):
        super().__init__(buckets_number, **kwargs)
        self._estimator = kmedoids.KMedoids(buckets_number, metric="euclidean")
        self._helper_imputer = SimpleImputer(strategy="median")

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:

        X_new = self._helper_imputer.fit_transform(X.filter(regex="^ft_num"))
        self._estimator = self._estimator.fit(X_new)

        self.centroids = self._estimator.cluster_centers_
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        X_num = X.filter(regex="^ft_num")
        X_copy = self._helper_imputer.transform(X_num)
        # TODO: Write a version with a weighted average based on distance space form transform function
        X_assignments = self._estimator.predict(X=X_copy)
        impute_values = self.centroids[X_assignments]

        X_new = pd.DataFrame(np.where(np.isnan(X_num), impute_values, X_num), X.index, X_num.columns)
        return replace_ft_num(X, X_new)

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

# TODO:
# Try these https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
# Especially, Spectral, DBSCAN, Agglomerative, BisectingKMeans
