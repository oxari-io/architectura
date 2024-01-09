from typing import Union
from sklearn.tree import DecisionTreeRegressor
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
from .core import BucketImputerBase, RegressionImputerBase
from enum import Enum, auto
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer


class MVEImputer(RegressionImputerBase):

    class Strategy(Enum):
        KNN = auto()
        BAYESRIDGE = auto()
        RIDGE = auto()
        DT = auto()

    def __init__(self, sub_estimator=Strategy.RIDGE.value, verbose=False, max_iter=15, internal_scaler=PowerTransformer(), **kwargs):
        super().__init__(internal_scaler=internal_scaler, **kwargs)
        self.sub_estimator = self._instantiate_strategy(sub_estimator) if isinstance(sub_estimator, self.Strategy) else sub_estimator
        self.verbose = verbose
        self._estimator = IterativeImputer(estimator=self.sub_estimator, verbose=self.verbose, max_iter=max_iter)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.logger.debug(f"Fitting {self.__class__.__name__} with {self.sub_estimator.__class__.__name__}")
        X_num = X.filter(regex='^ft_num')
        X_train_scaled = pd.DataFrame(self._fit_scaler(X_num, y), columns=X_num.columns, index=X_num.index)

        self._estimator = self._estimator.fit(X_train_scaled)
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        X_num = X.filter(regex='^ft_num')
        X_scaled_imputed = pd.DataFrame(self._scale_transform(X_num), index=X_num.index, columns=X_num.columns)
        X_new = pd.DataFrame(self._scaler.inverse_transform(X_scaled_imputed), index=X_num.index, columns=X_num.columns)
        return replace_ft_num(X, X_new)

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

    def _instantiate_strategy(self, strategy:Strategy):
        if self.Strategy.KNN == strategy:
            return KNeighborsRegressor()
        if self.Strategy.BAYESRIDGE == strategy:
            return BayesianRidge()
        if self.Strategy.RIDGE == strategy:
            return KernelRidge(kernel='rbf')
        if self.Strategy.DT == strategy:
            return DecisionTreeRegressor()

    def get_config(self):
        return {"strategy": self.sub_estimator.__class__.__name__, "imputer": f"{self.name}:{self.sub_estimator.__class__.__name__}", **super().get_config()}




class OldOxariImputer(MVEImputer):

    def __init__(self, **kwargs):
        super().__init__(sub_estimator=RandomForestRegressor(), **kwargs)


class GammaImputer(MVEImputer):

    def __init__(self, **kwargs):

        sub_estimator = make_pipeline([MinMaxScaler(), GammaRegressor()])
        super().__init__(sub_estimator=sub_estimator, **kwargs)