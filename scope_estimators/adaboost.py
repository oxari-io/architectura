from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from base import OxariScopeEstimator, OxariOptimizer
import numpy as np
import pandas as pd
import optuna
from base.oxari_types import ArrayLike
from base.metrics import optuna_metric
from base import OxariTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor
from typing_extensions import Self


class AdaboostOptimizer(OxariOptimizer):

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

    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):

        # n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.

        # https://medium.com/@chaudhurysrijani/tuning-of-adaboost-with-computational-complexity-8727d01a9d20
        param_space = {
            # 'estimator': trial.suggest_categorical('estimator', [DecisionTreeRegressor(), LinearRegression()]),
            'n_estimators': trial.suggest_categorical('n_estimators', [10, 50, 100, 500, 1000]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
        }

        model = AdaBoostRegressor(**param_space).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred)


class AdaboostEstimator(OxariScopeEstimator):

    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = AdaBoostRegressor()
        self._optimizer = optimizer or AdaboostOptimizer()
        #

    def fit(self, X, y=None, **kwargs) -> Self:
        self.n_features_in_ = X.shape[1]
        X_ = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self._estimator = self._estimator.set_params(**self.params).fit(X_, y.values.ravel())

        return self

    def predict(self, X) -> ArrayLike:
        return self._estimator.predict(X)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)

    def get_config(self, deep=True):
        return {**self._estimator.get_params(), **super().get_config(deep)}
