import itertools
from typing import Union
from base.common import OxariImputer
from base.dataset_loader import OxariDataManager
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from base.mappings import NumMapping
from base.metrics import mape, dunn_index
from sklearn import metrics
from sklearn import cluster
import kmedoids
from base.common import OxariEvaluator, DefaultClusterEvaluator



class KMeansBucketImputer(OxariImputer):
    def __init__(self, buckets_number=3, **kwargs):
        super().__init__(**kwargs)
        self.bucket_number = buckets_number
        self.list_of_skipped_columns = ['year', 'isin'] + NumMapping.get_targets()
        self.columns_to_fit = set(NumMapping.get_features()) - set(["revenue"])
        self.fallback_fallback_value = 0
        self._estimator = cluster.KMeans(self.bucket_number)
        self._helper_imputer = SimpleImputer(strategy="median")
        self._evaluator = DefaultClusterEvaluator()

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "OxariImputer":
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """
        self._helper_imputer = self._helper_imputer.fit(X)
        X_copy = self._helper_imputer.transform(X)
        self._estimator = self._estimator.fit(X_copy)
        

        self.centroids = self._estimator.cluster_centers_ 
        
        #TODO: Remove this line
        test = self.evaluate(X_copy, self._estimator.predict(X_copy))
        
        return self


    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_copy = self._helper_imputer.transform(X)
        # TODO: Write a version with a weighted average based on distance space form transform function
        X_assignments = self._estimator.predict(X=X_copy)
        impute_values = self.centroids[X_assignments]
        
        new_X = pd.DataFrame(np.where(np.isnan(X), impute_values, X) , X.index,X.columns)           
        return new_X
    
    def evaluate(self, X, labels, **kwargs):
        return super().evaluate(X, labels, **kwargs)
    

class KMedianBucketImputer(KMeansBucketImputer):
    def __init__(self, buckets_number=3, **kwargs):
        super().__init__(buckets_number, **kwargs)
        self._estimator = kmedoids.KMedoids(buckets_number, metric="euclidean")
        
# TODO:
# Try these https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
# Especially, Spectral, DBSCAN, Agglomerative, BisectingKMeans 