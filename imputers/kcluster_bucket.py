from typing import Union
from typing_extensions import Self
import kmedoids
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.impute import SimpleImputer, KNNImputer

from base.common import DefaultClusterEvaluator, OxariImputer
from base.mappings import NumMapping

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
        self._estimator = self._estimator.fit(X)
        # X_copy = self._estimator.transform(X)

        # self.centroids = self._estimator.cluster_centers_

        #TODO: Remove this line
        # test = self.evaluate(X_copy, self._estimator.predict(X_copy))

        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = self._estimator.transform(X)
        new_X = pd.DataFrame(new_X, X.index, X.columns)
        return new_X

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)


class KMedianBucketImputer(BucketImputerBase):
    def __init__(self, buckets_number=3, **kwargs):
        super().__init__(buckets_number, **kwargs)
        self._estimator = kmedoids.KMedoids(buckets_number, metric="euclidean")
        self._helper_imputer = SimpleImputer(strategy="median")

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "OxariImputer":
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """
        self._helper_imputer = self._helper_imputer.fit(X)
        X_copy = self._helper_imputer.transform(X)
        self._estimator = self._estimator.fit(X_copy)

        self.centroids = self._estimator.cluster_centers_

        #TODO: Remove this line
        # test = self.evaluate(X_copy, self._estimator.predict(X_copy))

        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_copy = self._helper_imputer.transform(X)
        # TODO: Write a version with a weighted average based on distance space form transform function
        X_assignments = self._estimator.predict(X=X_copy)
        impute_values = self.centroids[X_assignments]

        new_X = pd.DataFrame(np.where(np.isnan(X), impute_values, X), X.index, X.columns)
        return new_X

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

# TODO:
# Try these https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
# Especially, Spectral, DBSCAN, Agglomerative, BisectingKMeans
