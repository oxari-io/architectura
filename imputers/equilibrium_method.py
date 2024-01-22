# Input: X with col1, col2, col3



# 
# 

import random
import time
from typing import Union
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import column
from tqdm import tqdm
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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EquilibriumImputer(RegressionImputerBase):

    class Strategy(Enum):
        KNN = auto()
        BAYESRIDGE = auto()
        RIDGE = auto()
        DT = auto()



    def __init__(self, sub_estimator=Strategy.RIDGE, verbose=False, max_iter=100, diff_tresh=0.1, mims_tresh=0.01, max_diff_increase_thresh=0.5, internal_scaler=PowerTransformer(), **kwargs):
        super().__init__(internal_scaler=internal_scaler, **kwargs)
        self.i_counter:int = 0
        self.history_diffs = []
        self.history_mims = []
        self.history_counter = [0]
        self.max_iter = max_iter
        self.diff_tresh = diff_tresh
        self.mims_tresh = mims_tresh
        self.max_diff_increase_thresh = max_diff_increase_thresh
        self.skip_converged_cols = kwargs.get('skip_converged_cols', False)
        self.verbose = verbose
        self._sub_estimator = self._instantiate_strategy(sub_estimator) if isinstance(sub_estimator, self.Strategy) else sub_estimator
        self._estimator = IterativeImputer(estimator=self._sub_estimator, verbose=self.verbose)
        self.statistics = {}

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # # Initializing
        # For each col:
        #     train model to predict missing values
        start_time = time.time()

        self.logger.debug(f"Fitting {self.__class__.__name__} with {self._sub_estimator.__class__.__name__}")
        X_num = X.filter(regex='^ft_num')
        self._features_transformed = list(X_num.columns)

        X_train_scaled = pd.DataFrame(self._fit_scaler(X_num, y), columns=X_num.columns, index=X_num.index)

        self._estimator = self._estimator.fit(X_train_scaled)
        # For unified difference computation
        self._diff_scaler = MinMaxScaler().fit(X_train_scaled)

        self.statistics["fit_time"] = time.time() - start_time
        return self



    def transform(self, X, **kwargs) -> ArrayLike:
        # X'_0 = predict cols with trained model and fill only missing values

        # # Iterative part
        # While not converged:
        #     For each col:
        #         col_i = predict model(X'_i-1/col_i) and fill only missing values
        #     X'_i = [col_i for each col]
        start_time = time.time()

        X_num = X.filter(regex='^ft_num')
        X_num_missing_mask = X_num.isna()
        self.col_na_counts = X_num_missing_mask.sum()
        X_0 = pd.DataFrame(self._scale_transform(X_num), index=X_num.index, columns=X_num.columns)
        X_i = X_0.copy()
        X_j = None
        iter_diff = -1
        columns = list(X_num.columns)
        
        self.history_diffs = []
        self.history_mims = []
        self.history_counter = [0]        
        self.is_col_converged = np.ones(len(columns)) < 0
        pbar = tqdm(total=self.max_iter)
        while True:
            iter_diff = 0
            list_of_new_cols = []
            for idx, col in enumerate(columns):

                if (self.skip_converged_cols or kwargs.get("skip_converged_cols", False)) and (self.is_col_converged[idx] == True):
                    # Skip if converged
                    list_of_new_cols.append(X_i[col].copy())
                    continue
                X_temp = X_i.copy()
                X_temp[col] = np.where(X_num_missing_mask[col], np.nan, X_temp[col])
                X_temp_filled = pd.DataFrame(self._estimator.transform(X_temp), index=X_num.index, columns=X_num.columns)
                    
                list_of_new_cols.append(X_temp_filled[col].copy())
            X_j = pd.concat(list_of_new_cols, axis=1)

            self.history_diffs.append(self._compute_diffs(X_i, X_j))
            self.history_mims.append(self._compute_mims(X_i, X_j))
            self.history_counter.append(self._compute_counter(X_i, X_j))
            pbar.update(1)
            if self._is_converged(X_j, X_i):
                
                break

            X_i = X_j.copy()

        pbar.close()
            

        X_new = pd.DataFrame(self._scaler.inverse_transform(X_i), index=X_i.index, columns=X_i.columns).fillna(0)
        self.statistics["transform_time"] = time.time() - start_time
        return replace_ft_num(X, X_new)

    def _compute_counter(self, X_i, X_j):
        _,_ = X_i, X_j # Inputs not important here
        return self.history_counter[-1] + 1

    def _compute_diffs(self, X_i, X_j):
        iter_diffs = np.abs(self._diff_scaler.transform(X_i) - self._diff_scaler.transform(X_j))
        sumiter_diffs = np.sum(iter_diffs, axis=0)/self.col_na_counts.values
        return sumiter_diffs

    def _compute_mims(self, X_i, X_j):
        # Source: A Markov chain Monte Carlo algorithm for multiple imputation in large surveys, Daniel Schunk
        # Source: IMPUTATION OF THE 2002 WAVE OF THE SPANISH SURVEY OF HOUSEHOLD FINANCES (EFF), Cristina Barceló
        # Calculate the medians and interquartile ranges for X_t and X_t_minus_1
        M_y_t = np.median(X_j, axis=0)
        M_y_t_minus_1 = np.median(X_i, axis=0)
        IQ_R_y_t = np.subtract(*np.percentile(X_j, [75, 25], axis=0))
        IQ_R_y_t_minus_1 = np.subtract(*np.percentile(X_i, [75, 25], axis=0))

        # Calculate the convergence criterion
        median_diff = np.vstack([M_y_t , IQ_R_y_t]) - np.vstack([M_y_t_minus_1 , IQ_R_y_t_minus_1])
        diff_diag = np.diag(median_diff.T @ median_diff) # Only interested in the diagonals
        norm_diff = np.sqrt(diff_diag)  # Compute norms column-wise        
        return norm_diff



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
        """
        Calculate the convergence criterion for imputed data across iterations
        assuming that X_t and X_t_minus_1 are matrices where each column represents an independent y.

        Parameters:
        X_t (np.ndarray): The data at iteration t.
        X_t_minus_1 (np.ndarray): The data at iteration t-1.

        Returns:
        np.ndarray: The convergence criterion values for each variable (column).
        """

        if len(self.history_counter) < 5:
            return False
        

        # Stop if the counter reaches threshold
        if self.history_counter[-1] > self.max_iter:
            self.logger.info('Stopping - Maximum iterations reached')
            return True

        # Stop if all column differences are small enough
        is_diff_converged = self.history_diffs[-1] < self.diff_tresh
        is_all_diff_converged = np.all(is_diff_converged)
        if is_all_diff_converged:
            self.logger.info('Stopping - All diffs small enough')
            return True

        # If many of differences are increasing stop
        is_diff_increase = self.history_diffs[-1] > self.history_diffs[-2]
        is_all_diff_increased = np.mean(is_diff_increase) > self.max_diff_increase_thresh
        if is_all_diff_increased:
            self.logger.info('Stopping - many diffs increased')
            return True


        # Stop if all mims are small enough
        self.is_col_converged = self.history_mims[-1] < self.mims_tresh
        is_all_converged = np.all(self.is_col_converged)
        
        if is_all_converged:
            self.logger.info('Stopping - All mims small enough')
            return True
        return False


    def evaluate(self, X, y=None, **kwargs):
        return super().evaluate(X, y, **kwargs)

    def get_config(self):
        return {"strategy": self._sub_estimator.__class__.__name__, 
                "max_iter":self.max_iter,
                "completed_iter":self.history_counter[-1],
                "diff_tresh":self.diff_tresh,
                "mims_tresh":self.mims_tresh,
                "max_diff_increase_thresh":self.max_diff_increase_thresh,
                "statistics":dict(self.statistics),
                "imputer": f"{self.name}:{self._sub_estimator.__class__.__name__}", "skip_cols":self.skip_converged_cols, "final_iter":self.i_counter, **super().get_config()}

    def visualize(self):
        if not(len(self.history_diffs) and len(self.history_mims)):
            raise Exception("You need to run transform or evaluate at least once")

        diffs = pd.DataFrame(np.vstack(self.history_diffs), columns=self._features_transformed)
        mimss = pd.DataFrame(np.vstack(self.history_mims), columns=self._features_transformed)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        sns.lineplot(data=diffs.reset_index().melt('index', var_name='Feature'), x='index', y='value', hue='Feature', ax=axes[0])
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Feature differences')
        sns.lineplot(data=mimss.reset_index().melt('index', var_name='Feature'), x='index', y='value', hue='Feature', ax=axes[1])
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Distributional shift')
        fig.tight_layout()
        return fig, axes

class FastEquilibriumImputer(EquilibriumImputer):

    def __init__(self, sub_estimator=EquilibriumImputer.Strategy.RIDGE, verbose=False, max_iter=100, diff_tresh=0.1, mims_tresh=0.01, max_diff_increase_thresh=0.5, internal_scaler=PowerTransformer(), **kwargs):
        super().__init__(sub_estimator, verbose, max_iter, diff_tresh, mims_tresh, max_diff_increase_thresh, internal_scaler, **kwargs)

    def transform(self, X:pd.DataFrame, **kwargs) -> ArrayLike:
        start_time = time.time()


        X_num = X.filter(regex='^ft_num')
        X_num_missing_mask = X_num.isna()
        self.col_na_counts = X_num_missing_mask.sum()

        X_0 = pd.DataFrame(self._scale_transform(X_num), index=X_num.index, columns=X_num.columns)
        X_i = X_0.copy()
        X_j = None
        iter_diff = -1
        columns = list(X_num.columns)
        
        self.history_diffs = []
        self.history_mims = []
        self.history_counter = [0]        
        self.is_col_converged = np.ones(len(columns)) < 0
        pbar = tqdm(total=self.max_iter)
        is_not_converged = True
        while is_not_converged:
            average_diffs = []
            average_mimss = []
            indices = np.random.permutation(list(range(len(columns))))
            X_old = X_i.copy()
            for idx in indices:
                col = columns[idx]
                if (self.skip_converged_cols or kwargs.get("skip_converged_cols", False)) and (self.is_col_converged[idx] == True):
                    # Skip if converged
                    average_diffs.append(0)
                    average_mimss.append(0)
                    X_j = X_i.copy()
                    continue
                X_temp = X_i.copy()
                X_temp[col] = np.where(X_num_missing_mask[col], np.nan, X_temp[col])
                X_j = pd.DataFrame(self._estimator.transform(X_temp), index=X_num.index, columns=X_num.columns)
                    
                # average_diffs.append(np.sum(self._compute_diffs(X_i, X_j)))
                # average_mimss.append(np.sum(self._compute_mims(X_i, X_j)))
 
                X_i = X_j.copy()
            
            self.history_diffs.append(self._compute_diffs(X_old, X_j))
            self.history_mims.append(self._compute_mims(X_old, X_j))
            self.history_counter.append(self._compute_counter(X_old, X_j))
            pbar.update(1)
            is_not_converged = not self._is_converged(X_j, X_old)

        pbar.close()
            

        X_new = pd.DataFrame(self._scaler.inverse_transform(X_i), index=X_i.index, columns=X_i.columns).fillna(0)
        self.statistics["transform_time"] = time.time() - start_time
        return replace_ft_num(X, X_new)