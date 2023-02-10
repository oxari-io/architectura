import itertools
from typing import Union
from typing_extensions import Self
import kmedoids
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.impute import SimpleImputer, KNNImputer
from .core import BucketImputerBase
from autoimpute.imputations import MiceImputer, MultipleImputer
from enum import Enum

class AutoImputer(BucketImputerBase):
    class strategies(Enum):
        NORM="norm"
        NOCB="nocb"
        LOCF="locf"
        LSQ = "least squares"
        RANDOM = "random"
        LINEAR = "linear"
        BAYES = "bayesian least squares"
        INTERPOLATE = "interpolate"
        MEAN = "univariate default"
        DEFAULT = None
        P_DEFAULT = "predictive default"
        PMM = "pmm"
        LRD = "lrd"

    def __init__(self, strategy:strategies=strategies.DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self._estimator = MultipleImputer(strategy==self.strategy)


    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """
        self._estimator = self._estimator.fit(X)
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = self._estimator.transform(X)
        X_new = pd.DataFrame(X_new, index=X.index, columns=X.columns)
        return X_new

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

    def get_config(self):
        return {"strategy": self.strategy, **super().get_config()}