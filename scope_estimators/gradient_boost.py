from typing import Union
from base import OxariScopeEstimator, OxariRegressor, OxariMixin
import numpy as np
import pandas as pd
from base.oxari_types import ArrayLike
import xgboost as xgb
import sklearn
#do I want that? I do, right?
from xgboost import XGBRegressor


class XGBRegressor(OxariRegressor):
    
    # n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
    # max_depth: The maximum depth of each tree, often values are between 1 and 10.
    # eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
    # subsample: The number of samples (rows) used in each tree, set to a value between 0 and 1, often 1.0 to use all samples.
    # colsample_bytree: Number of features (columns) used in each tree, set to a value between 0 and 1, often 1.0 to use all features.
    # current values are taken from machinelearningmastery tutorial
    def __init__(self, n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, **kwargs) -> None:
        super().__init__(**kwargs)
        self._regressor = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, eta=eta, subsample=subsample, colsample_bytree=colsample_bytree)

    # we can add more parameters if we want
    def fit(self, X, y, **kwargs) -> "OxariRegressor":
        return self.fit(X, y)
    
    def predict(self, X:ArrayLike, **kwargs) -> ArrayLike:
        return self.predict(X)

    # alternative to "def _set_meta"
    def set_params(self, X:ArrayLike, **kwargs) -> ArrayLike:
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)

    """other potentially useful functions:
        - evals_result()
        - save_model(fname)
        - score(X, y, sample_weight=None)"""

