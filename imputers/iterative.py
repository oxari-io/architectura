from typing import Union
from typing_extensions import Self
from sklearn.experimental import enable_iterative_imputer
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from base.common import DefaultClusterEvaluator, OxariImputer
from base.helper import replace_ft_num
from base.mappings import NumMapping
from sklearn.linear_model import BayesianRidge, Ridge, GammaRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from base.oxari_types import ArrayLike
from .core import BucketImputerBase
from enum import Enum
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


class MVEImputer(OxariImputer):

    class strategies(Enum):
        BAYESRIDGE = BayesianRidge()
        RIDGE = KernelRidge(kernel='rbf')
        KNN = KNeighborsRegressor()

    def __init__(self, sub_estimator=strategies.RIDGE.value, verbose=False, max_iter=10, **kwargs):
        super().__init__(**kwargs)
        self.sub_estimator = sub_estimator
        self.verbose = verbose
        self._estimator = IterativeImputer(estimator=self.sub_estimator, verbose=self.verbose, max_iter=max_iter)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.logger.debug(f"Fitting {self.__class__.__name__} with {self.sub_estimator.__class__.__name__}")
        self._estimator = self._estimator.fit(X.filter(regex='^ft_num'))
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        X_num = X.filter(regex='^ft_num')
        X_new = self._estimator.transform(X_num)
        X_new = pd.DataFrame(X_new, index=X_num.index, columns=X_num.columns)
        return replace_ft_num(X, X_new)

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

    def get_config(self):
        return {"strategy": self.sub_estimator.__class__.__name__, "imputer": f"{self.name}:{self.sub_estimator.__class__.__name__}", **super().get_config()}


class OldOxariImputer(MVEImputer):

    def __init__(self, **kwargs):
        super().__init__(sub_estimator=RandomForestRegressor(), **kwargs)


class GammaImputer(MVEImputer):

    def __init__(self, **kwargs):

        sub_estimator = make_pipeline([MinMaxScaler(), GammaRegressor()])
        super().__init__(sub_estimator=sub_estimator, **kwargs)