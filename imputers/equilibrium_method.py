# Input: X with col1, col2, col3



# 
# 

import random
from typing import Union
from sklearn.base import BaseEstimator
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


class EquilibriumImputer(OxariImputer):

    class strategies(Enum):
        # BAYESRIDGE = BayesianRidge()
        RIDGE = KernelRidge(kernel='rbf')
        KNN = KNeighborsRegressor()
        RF = RandomForestRegressor(max_depth=5)

    def __init__(self, sub_estimator=strategies.RIDGE.value, verbose=False, max_iter=100, **kwargs):
        super().__init__(**kwargs)
        self.i_counter:int = 0
        self.diff_history = [999]
        self.sub_estimator = sub_estimator
        self.max_iter = max_iter
        self.verbose = verbose
        self._estimator = IterativeImputer(estimator=self.sub_estimator, verbose=self.verbose)

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
        X_0 = pd.DataFrame(self._estimator.transform(X_num), index=X_num.index, columns=X_num.columns)
        X_i = X_0.copy()
        X_j = None
        iter_diff = -1
        columns = list(X_num.columns)
        

        while not self._is_converged(iteration_diff=iter_diff):
            iter_diff = 0
            list_of_new_cols = []
            for col in columns:
                X_temp = X_i.copy()
                X_temp[col] = np.where(X_num_missing_mask[col], np.nan, X_temp[col])
                X_temp_filled = pd.DataFrame(self._estimator.transform(X_temp), index=X_num.index, columns=X_num.columns)
                
                list_of_new_cols.append(X_temp_filled[col].copy())
            X_j = pd.concat(list_of_new_cols, axis=1)
            # We can take the difference of the entire table as the non-missing fields do not change and therefore not contirbute to the sum
            iter_diff = np.sum(np.sum(np.abs(X_i - X_j)))
            X_i = X_j.copy()
            self.diff_history.append(iter_diff)

                
            

        X_new = X_i
        return replace_ft_num(X, X_new)

    def _is_converged(self, **kwargs):
        # Equilibrium: The difference between predictions at the current step vs the last step
        if self.diff_history[-1] == 0:
            return True
        if self.i_counter > self.max_iter:
            return True
        self.i_counter +=1
        return False

    # def _is_converged(self, **kwargs):
    #     # Equilibrium: The difference between predictions at the current step vs the last step
    #     premediate_difference = np.abs(self.diff_history[-2]-self.diff_history[-3])
    #     immediate_difference = np.abs(self.diff_history[-1]-self.diff_history[-2])
    #     if premediate_difference:
    #         return True
    #     self.i_counter +=1
    #     return False

    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

    def get_config(self):
        return {"strategy": self.sub_estimator.__class__.__name__, "imputer": f"{self.name}:{self.sub_estimator.__class__.__name__}-{self.max_iter}", "final_iter":self.i_counter, **super().get_config()}
