# Input: X with col1, col2, col3



# 
# 

import random
from typing import Union
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import column
from typing_extensions import Self
# noqa
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


class EquilibriumImputer(RegressionImputerBase):

    class Strategy(Enum):
        KNN = auto()
        BAYESRIDGE = auto()
        RIDGE = auto()
        DT = auto()



    def __init__(self, sub_estimator=Strategy.RIDGE, verbose=False, max_iter=100, internal_scaler=PowerTransformer(), **kwargs):
        super().__init__(internal_scaler=internal_scaler, **kwargs)
        self.i_counter:int = 0
        self.diff_history = []
        self.max_iter = max_iter
        self.verbose = verbose
        self._sub_estimator = self._instantiate_strategy(sub_estimator) if isinstance(sub_estimator, self.Strategy) else sub_estimator
        self._estimator = IterativeImputer(estimator=self._sub_estimator, verbose=self.verbose)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # # Initializing
        # For each col:
        #     train model to predict missing values
        
        self.logger.debug(f"Fitting {self.__class__.__name__} with {self._sub_estimator.__class__.__name__}")
        X_num = X.filter(regex='^ft_num')

        X_train_scaled = pd.DataFrame(self._fit_scaler(X_num, y), columns=X_num.columns, index=X_num.index)

        self._estimator = self._estimator.fit(X_train_scaled)
        # For unified difference computation
        self._diff_scaler = MinMaxScaler().fit(X_train_scaled)
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
        X_0 = pd.DataFrame(self._scale_transform(X_num), index=X_num.index, columns=X_num.columns)
        X_i = X_0.copy()
        X_j = None
        iter_diff = -1
        columns = list(X_num.columns)
        

        while True:
            iter_diff = 0
            list_of_new_cols = []
            for col in columns:
                X_temp = X_i.copy()
                X_temp[col] = np.where(X_num_missing_mask[col], np.nan, X_temp[col])
                X_temp_filled = pd.DataFrame(self._estimator.transform(X_temp), index=X_num.index, columns=X_num.columns)
                
                list_of_new_cols.append(X_temp_filled[col].copy())
            X_j = pd.concat(list_of_new_cols, axis=1)
            # We can take the difference of the entire table as the non-missing fields do not change and therefore not contirbute to the sum
            iter_diff = np.sum(np.sum(np.abs(self._scaler.transform(X_i) - self._scaler.transform(X_j))))
            # iter_diff = np.sum(np.sum(np.abs(X_i-X_j)))
            self.diff_history.append(iter_diff)
            if self._is_converged(X_j, X_i):
                break

            X_i = X_j.copy()

                
            

        X_new = pd.DataFrame(self._scaler.inverse_transform(X_i), index=X_i.index, columns=X_i.columns) 
        return replace_ft_num(X, X_new)



    def _instantiate_strategy(self, strategy:Strategy):
        if self.Strategy.KNN == strategy:
            return KNeighborsRegressor(7)
        if self.Strategy.BAYESRIDGE == strategy:
            return BayesianRidge()
        if self.Strategy.RIDGE == strategy:
            return KernelRidge(kernel='rbf')
        if self.Strategy.DT == strategy:
            return DecisionTreeRegressor()


    def _is_converged(self, X_t, X_t_minus_1, **kwargs):
        # Equilibrium: The difference between predictions at the current step vs the last step
        # Have at least two runs

        # if len(self.diff_history) < 3:
        #     return False

        # # No change stop
        # if self.diff_history[-1] <= 0:
        #     self.logger.info('Stopping - No change')
        #     return True


        # # Small change stop
        # if self.diff_history[-1] <= 1e-10:
        #     self.logger.info('Stopping - Small change')
        #     return True
        
        # # Small relative change stop
        # if (np.abs(self.diff_history[-1]-self.diff_history[-2])/self.diff_history[-2]) <= 1e-5:
        #     self.logger.info('Stopping - Small relative change')
        #     return True

        # # Have at least two runs
        # if len(self.diff_history) < 5:
        #     return False

        # # Consecutive increases
        # if self.diff_history[-1] > self.diff_history[-2] > self.diff_history[-3] > self.diff_history[-4]:
        #     self.logger.info('Stopping - Increasing')
        #     return True

        if self.i_counter > self.max_iter:
            self.logger.info('Stopping - Maximum iterations reached')
            return True
        self.i_counter +=1
        return False

    def _is_converged(self, X_t, X_t_minus_1, **kwargs):
        """
        Calculate the convergence criterion for imputed data across iterations
        assuming that X_t and X_t_minus_1 are matrices where each column represents an independent y.

        Parameters:
        X_t (np.ndarray): The data at iteration t.
        X_t_minus_1 (np.ndarray): The data at iteration t-1.

        Returns:
        np.ndarray: The convergence criterion values for each variable (column).
        """
        # Calculate the medians and interquartile ranges for X_t and X_t_minus_1
        M_y_t = np.median(X_t, axis=0)
        M_y_t_minus_1 = np.median(X_t_minus_1, axis=0)
        IQ_R_y_t = np.subtract(*np.percentile(X_t, [75, 25], axis=0))
        IQ_R_y_t_minus_1 = np.subtract(*np.percentile(X_t_minus_1, [75, 25], axis=0))

        # Calculate the convergence criterion
        median_diff = (M_y_t / IQ_R_y_t) - (M_y_t_minus_1 / IQ_R_y_t_minus_1)
        norm_diff = np.linalg.norm(median_diff, axis=0)  # Compute norms column-wise

        return norm_diff

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
        return {"strategy": self._sub_estimator.__class__.__name__, "imputer": f"{self.name}:{self._sub_estimator.__class__.__name__}-{self.max_iter}", "final_iter":self.i_counter, **super().get_config()}
