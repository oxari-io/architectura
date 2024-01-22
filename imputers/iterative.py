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
from base.helper import convert_to_df, replace_ft_num
from base.mappings import NumMapping
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge, GammaRegressor
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
        X_train_scaled = convert_to_df(self._fit_scaler(X_num, y), X_num)

        self._estimator = self._estimator.fit(X_train_scaled)
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        X_num = X.filter(regex='^ft_num')
        X_scaled_imputed = convert_to_df(self._scale_transform(X_num), X_num) 
        X_new = convert_to_df(self._scaler.inverse_transform(X_scaled_imputed), X_num).fillna(0).replace([np.inf], np.finfo(np.float64).max)
        return replace_ft_num(X, X_new)

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

    def _instantiate_strategy(self, strategy:Strategy):
        if self.Strategy.KNN == strategy:
            return KNeighborsRegressor(9)
        if self.Strategy.RIDGE == strategy:
            return KernelRidge(kernel='rbf')
        if self.Strategy.DT == strategy:
            return DecisionTreeRegressor()

    def get_config(self, deep=True):
        return {"strategy": self.sub_estimator.__class__.__name__, "imputer": f"{self.name}:{self.sub_estimator.__class__.__name__}", **super().get_config(deep)}




class OldOxariImputer(MVEImputer):

    def __init__(self, **kwargs):
        super().__init__(sub_estimator=RandomForestRegressor(n_estimators=50), **kwargs)


class GammaImputer(MVEImputer):

    def __init__(self, **kwargs):

        sub_estimator = make_pipeline([MinMaxScaler(), GammaRegressor()])
        super().__init__(sub_estimator=sub_estimator, **kwargs)