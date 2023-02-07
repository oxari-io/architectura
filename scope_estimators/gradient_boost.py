from typing import Union
from base import OxariScopeEstimator, OxariRegressor, OxariMixin, OxariOptimizer
import numpy as np
import pandas as pd
import optuna
from base.oxari_types import ArrayLike
from base.metrics import optuna_metric
import xgboost as xgb
import sklearn
#do I want that? I do, right?
from xgboost import XGBRegressor

""" Relatively useful in hyperparameter tuning: 
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
"""

class XGBOptimizer(OxariOptimizer):
    # n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
    # max_depth: The maximum depth of each tree, often values are between 1 and 10.
    # eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
    # gamma = 0 : A smaller value like 0.1-0.2 can also be chosen for starting. This will anyways be tuned later.
    # subsample: The number of samples (rows) used in each tree, set to a value between 0 and 1, often 1.0 to use all samples.
    # colsample_bytree: Number of features (columns) used in each tree, set to a value between 0 and 1, often 1.0 to use all features.
    # objective: determines the loss function to be used
    def __init__(self, n_estimators=1000, max_depth=5, eta=0.1, min_child_weight=1, gamma=0, subsample=0.8, 
                 colsample_bytree=0.8, objective= 'reg:squarederror', scale_pos_weight=1, seed=27, **kwargs) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            eta=eta,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight, #this may need to be readjusted if this is not a highly imbalanced class problem
            gamma=gamma,
            objective=objective,
            scale_pos_weight=scale_pos_weight,
            seed=seed,
            **kwargs,
        )
    
    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Explore the hyperparameter tning space with optuna.
        Creates csv and pickle files with the saved hyperparameters for classification

        Parameters:
        X_train (numpy array): training data (features)
        y_train (numpy array): training data (targets)
        X_val (numpy array): validation data (features)
        y_val (numpy array): validation data (targets)
        num_startup_trials (int): 
        n_trials (int): 

        Return:
        study.best_params (data structure): contains the best found parameters within the given space
        """

        # create optuna study
        # num_startup_trials is the number of random iterations at the beginiing
        study = optuna.create_study(
            study_name=f"xgboost_process_hp_tuning",
            direction="minimize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return study.best_params, df

    # TODO: Find better optimization ranges for the GaussianProcessEstimator
    def score_trial(self, trial:optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        # epsilon = trial.suggest_float("epsilon", 0.01, 0.2)
        # C = trial.suggest_float("C", 0.01, 2.0)
        eta = trial.suggest_float("eta", 0.01, 0.2)
        reg_alpha = trial.suggest_float("reg_alpha", 0.01, 0.2)
        
            
        max_size = len(X_train)
        sample_size = int(max_size*0.1)
        indices = np.random.randint(0, max_size, sample_size)
        model = XGBRegressor(eta=eta, reg_alpha=reg_alpha).fit(X_train.iloc[indices], y_train.iloc[indices])
        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred)




class XGBEstimator(OxariScopeEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = XGBRegressor()
        self._optimizer = optimizer or XGBOptimizer()

    def fit(self, X, y, **kwargs) -> "XGBEstimator":
        max_size = len(X)
        sample_size = int(max_size*0.1)
        indices = np.random.randint(0, max_size, sample_size)   
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self._estimator = self._estimator.set_params(**self.params).fit(X.iloc[indices], y.iloc[indices].values.ravel())
        # self.coef_ = self._estimator.coef_
        return self
       
    def predict(self, X) -> ArrayLike:
        return self._estimator.predict(X)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)     

    def check_conformance(self):
        pass

    def get_config(self, deep=True):
        return {**self._estimator.get_params(), **super().get_config(deep)}
        
       

    # # we can add more parameters if we want
    # def fit(self, X, y, **kwargs) -> "OxariRegressor":
    #     return self.fit(X, y)
    
    # def predict(self, X:ArrayLike, **kwargs) -> ArrayLike:
    #     return self.predict(X)

    # # alternative to "def _set_meta"
    # def set_params(self, X:ArrayLike, **kwargs) -> ArrayLike:
    #     self.feature_names_in_ = list(X.columns)
    #     self.n_features_in_ = len(self.feature_names_in_)