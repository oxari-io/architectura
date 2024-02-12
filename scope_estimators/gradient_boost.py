from lightgbm import LGBMRegressor
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from typing_extensions import Self
from base import OxariScopeEstimator
from base.common import OxariOptimizer
from base.metrics import optuna_metric
from base.oxari_types import ArrayLike
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

    def __init__(self, n_trials=10, n_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__(
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
            sampler=sampler,
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
        study:optuna.Study = optuna.create_study(
            study_name=f"xgboost_process_hp_tuning",
            direction="minimize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return study.best_params, df

    def score_trial(self, trial:optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        
        # max_depth: The maximum depth of each tree, often values are between 1 and 10.
        # colsample_bytree: Number of features (columns) used in each tree, set to a value between 0 and 1, often 1.0 to use all features.
        # gamma = 0 : A smaller value like 0.1-0.2 can also be chosen for starting. This will anyways be tuned later.
        # subsample: The number of samples (rows) used in each tree, set to a value between 0 and 1, often 1.0 to use all samples.
        # objective: determines the loss function to be used
        # eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
        # n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.

        param_space = {
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, step=100),
            }
        
            
        model = XGBRegressor(**param_space).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return self.metric(y_val, y_pred)


class LGBOptimizer(XGBOptimizer):
    def score_trial(self, trial:optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        
        param_space = {
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, 100),
            }
        
            
        model = LGBMRegressor(**param_space).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return self.metric(y_val, y_pred)    

class SklearnGBOptimizer(XGBOptimizer):
    def score_trial(self, trial:optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        
        param_space = {
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, 100),
            }
        
            
        model = GradientBoostingRegressor(**param_space).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return self.metric(y_val, y_pred)    



class XGBEstimator(OxariScopeEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator:XGBRegressor = XGBRegressor()
        self.set_optimizer(optimizer or XGBOptimizer())

    def fit(self, X, y, **kwargs) -> Self:
        self.n_features_in_ = X.shape[1]
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self._estimator = self._estimator.set_params(**self.params).fit(X, y.values.ravel())
        # self.coef_ = self._estimator.coef_
        return self
       
    def predict(self, X) -> ArrayLike:
        return self._estimator.predict(X)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)     

    def get_config(self, deep=True):
        return {**self._estimator.get_params(), **super().get_config(deep)}

class LGBEstimator(XGBEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = LGBMRegressor()
        self.set_optimizer(optimizer or LGBOptimizer())

class SklearnGBEstimator(XGBEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = GradientBoostingRegressor()
        self.set_optimizer(optimizer or SklearnGBOptimizer())

        
       

    # # we can add more parameters if we want
    # def fit(self, X, y, **kwargs) -> "OxariRegressor":
    #     return self.fit(X, y)
    
    # def predict(self, X:ArrayLike, **kwargs) -> ArrayLike:
    #     return self.predict(X)

    # # alternative to "def _set_meta"
    # def set_params(self, X:ArrayLike, **kwargs) -> ArrayLike:
    #     self.feature_names_in_ = list(X.columns)
    #     self.n_features_in_ = len(self.feature_names_in_)