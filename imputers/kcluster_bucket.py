from typing import Union
from typing_extensions import Self
import kmedoids
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.impute import SimpleImputer, KNNImputer

from base.common import DefaultClusterEvaluator, OxariImputer
from base.helper import convert_to_df, replace_ft_num
from base.mappings import NumMapping
from base.oxari_types import ArrayLike

from .core import BucketImputerBase, RegressionImputerBase


class KNNBucketImputer(RegressionImputerBase, BucketImputerBase):
    def __init__(self, num_buckets=3, **kwargs):
        super().__init__(**kwargs)
        self.bucket_number = num_buckets
        self._estimator = KNNImputer(n_neighbors=self.bucket_number)


    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """
        X_num = X.filter(regex="^ft_num")
        X_scaled = self._fit_scaler(X_num, y)
        self._estimator = self._estimator.fit(X_scaled)
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        X_num = X.filter(regex="^ft_num")
        X_scaled = convert_to_df(self._scale_transform(X_num), X_num) 
        X_new_reversed = convert_to_df(self._scaler.inverse_transform(X_scaled), X_num).fillna(0)
        return replace_ft_num(X, X_new_reversed)

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)


class KMeansBucketImputer(RegressionImputerBase, BucketImputerBase):
    def __init__(self, num_buckets=3, **kwargs):
        super().__init__(bucket_number=num_buckets, **kwargs)
        self.bucket_number = num_buckets        
        self._estimator = cluster.KMeans(num_buckets, n_init='auto')
        self._helper_imputer = SimpleImputer(strategy="median", keep_empty_features=True)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        X_num = X.filter(regex="^ft_num")
        X_new = self._helper_imputer.fit_transform(X_num)
        X_new_scaled = self._fit_scaler(convert_to_df(X_new, X_num), y)
        self._estimator = self._estimator.fit(X_new_scaled.values)
        
        self.medoids = convert_to_df(X_new_scaled, X_num)
        self.centroids = self._estimator.cluster_centers_
        return self

    def transform(self, X:pd.DataFrame, **kwargs) -> ArrayLike:
        X_num = X.filter(regex="^ft_num")
        X_msno_mask = X_num.isna()
        X_copy = convert_to_df(self._helper_imputer.transform(X_num), X_num)
        X_scaled = self._scaler.transform(X_copy) 

        X_assignments = self._estimator.predict(X=X_scaled)
        X_impute_values = self.centroids[X_assignments]
        
        X_imputed = convert_to_df(np.where(X_msno_mask, X_impute_values, X_scaled), X_num)
        X_new = convert_to_df(self._scaler.inverse_transform(X_imputed), X_num).fillna(0)
        return replace_ft_num(X, X_new)

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)


class KMedianBucketImputer(RegressionImputerBase, BucketImputerBase):
    def __init__(self, num_buckets=3, **kwargs):
        super().__init__(bucket_number=num_buckets, **kwargs)
        self.bucket_number = num_buckets        
        self._estimator = kmedoids.KMedoids(num_buckets, metric="euclidean")
        self._helper_imputer = SimpleImputer(strategy="median", keep_empty_features=True)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        X_num = X.filter(regex="^ft_num")
        X_new = self._helper_imputer.fit_transform(X_num)
        X_new_scaled = self._fit_scaler(convert_to_df(X_new, X_num), y)
        self._estimator = self._estimator.fit(X_new_scaled.values)
        
        self.medoids = convert_to_df(X_new_scaled, X_num)
        self.centroids = self._estimator.cluster_centers_
        return self

    def transform(self, X:pd.DataFrame, **kwargs) -> ArrayLike:
        X_num = X.filter(regex="^ft_num")
        X_msno_mask = X_num.isna()
        X_copy = convert_to_df(self._helper_imputer.transform(X_num), X_num)
        X_scaled = self._scaler.transform(X_copy) 

        X_assignments = self._estimator.predict(X=X_scaled)
        X_impute_values = self.medoids.values[X_assignments]
        
        X_imputed = convert_to_df(np.where(X_msno_mask, X_impute_values, X_scaled), X_num)
        X_new = convert_to_df(self._scaler.inverse_transform(X_imputed), X_num).fillna(0)
        return replace_ft_num(X, X_new)

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

# TODO:
# Try these https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
# Especially, Spectral, DBSCAN, Agglomerative, BisectingKMeans
