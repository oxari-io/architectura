# Input: X with col1, col2, col3



# 
# 

import random
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


class Equilibrium(OxariImputer):
    i_counter:int

    class strategies(Enum):
        BAYESRIDGE = BayesianRidge()
        RIDGE = KernelRidge(kernel='rbf')
        KNN = KNeighborsRegressor()

    def __init__(self, sub_estimator=strategies.RIDGE.value, verbose=False, max_iter=15, **kwargs):
        super().__init__(**kwargs)
        self.sub_estimator = sub_estimator
        self.verbose = verbose
        self._estimator = IterativeImputer(estimator=self.sub_estimator, verbose=self.verbose, max_iter=max_iter)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # # Initializing
        # For each col:
        #     train model to predict missing values
        
        self.logger.debug(f"Fitting {self.__class__.__name__} with {self.sub_estimator.__class__.__name__}")
        self._estimator = self._estimator.fit(X.filter(regex='^ft_num'))
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        # X'_0 = predict cols with trained model and fill only missing values

        # # Iterative part
        # While not converged:
        #     For each col:
        #         col_i = predict model(X'_i-1/col_i) and fill only missing values
        #     X'_i = [col_i for each col]

        X_num = X.filter(regex='^ft_num')

        X_num_missing_mask = X_num.isna()
        X_0 = self._estimator.transform(X_num)
        X_i = X_0.copy()
        iter_diff = -1
        columns = list(X_num.columns)
        

        while not self._is_converged(iteration_diff=iter_diff):
            iter_diff = 0
            columns = np.random.permutation(columns)
            for col in columns:
                other_cols = set(X_num.columns) - set([col]) # Not sure if needed
                X_temp = X_i.copy()
                X_temp[col] = np.where(X_num_missing_mask[col], np.nan, X_temp[col])
                X_j = self._estimator.transform(X_temp)
                # We can take the difference of the entire column as the non-missing fields do not change and therefore not contirbute to the sum
                col_abs_diff = np.abs(X_i[col] - X_j[col])
                col_sum_diff = col_abs_diff.sum()
                iter_diff += col_sum_diff
                X_i = X_j.copy()
            

        X_new = pd.DataFrame(X_i, index=X_num.index, columns=X_num.columns)
        return replace_ft_num(X, X_new)

    def _is_converged(self, **kwargs):
        # Equilibrium: The difference between predictions at the current step vs the last step

        if self.i_counter > 10:
            return True
        self.i_counter +=1
        return False

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

    def get_config(self):
        return {"strategy": self.sub_estimator.__class__.__name__, "imputer": f"{self.name}:{self.sub_estimator.__class__.__name__}", **super().get_config()}
