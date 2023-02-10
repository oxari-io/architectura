from typing import Union
from typing_extensions import Self
from sklearn.experimental import enable_iterative_imputer
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from base.common import DefaultClusterEvaluator, OxariImputer
from base.mappings import NumMapping
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from .core import BucketImputerBase
from enum import Enum


class MVEImputer(OxariImputer):

    class strategies(Enum):
        BAYESRIDGE = BayesianRidge()
        RIDGE = Ridge()
        NYSTROEM = Nystroem()
        KNN = KNeighborsRegressor()

    def __init__(self, sub_estimator = strategies.RIDGE.value, verbose=False, max_iter=10, **kwargs):
        super().__init__(**kwargs)
        self.sub_estimator = sub_estimator
        self.verbose = verbose
        self._estimator = IterativeImputer(estimator=self.sub_estimator, verbose=self.verbose, max_iter=max_iter)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.logger.debug(f"Fitting {self.__class__.__name__} with {self.sub_estimator.__class__.__name__}")
        self._estimator = self._estimator.fit(X)
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = self._estimator.transform(X)
        X_new = pd.DataFrame(X_new, index=X.index, columns=X.columns)
        return X_new

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

    def get_config(self):
        return {"strategy": self.sub_estimator.__class__.__name__, **super().get_config()}


class OldOxariImputer(MVEImputer):

    def __init__(self, **kwargs):
        super().__init__(sub_estimator=RandomForestRegressor(), **kwargs)