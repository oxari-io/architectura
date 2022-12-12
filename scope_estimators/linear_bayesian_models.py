from typing import Union
from base import OxariScopeEstimator, ReducedDataMixin
from base.common import OxariOptimizer
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
import optuna
from pmdarima.metrics import smape
from sklearn.metrics import mean_tweedie_deviance, mean_absolute_error
from sklearn import linear_model
from .linear.helper import PolynomialFeaturesMixin


class BayesianRegressorOptimizer(PolynomialFeaturesMixin, ReducedDataMixin, OxariOptimizer):
    def __init__(self, num_trials=50, num_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__(
            num_trials=num_trials,
            num_startup_trials=num_startup_trials,
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
            study_name=f"{self.__class__.__name__}_process_hp_tuning",
            direction="minimize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.num_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return study.best_params, df

    def score_trial(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train, X_val, y_val, **kwargs):
        alpha_1 = trial.suggest_float("alpha_init", 1e-8, 1, log=True)
        lambda_1 = trial.suggest_float("lambda_init", 1e-8, 1, log=True)
        n_iter = trial.suggest_int("n_iter", 100, 500, step=100)
        degree = trial.suggest_int("degree", 1, 10)
        X_train, y_train = self.polynomializer.set_params(degree=degree).fit_transform(X_train), y_train.values
        indices = self.get_sample_indices(X_train)
        X_val = self.polynomializer.set_params(degree=degree).fit_transform(X_val)
        model = linear_model.BayesianRidge(n_iter=n_iter, alpha_init=alpha_1, lambda_init=lambda_1).fit(X_train[indices], y_train[indices])
        y_pred = model.predict(X_val)

        return smape(y_true=y_val, y_pred=y_pred)


class BayesianRegressionEstimator(PolynomialFeaturesMixin, ReducedDataMixin, OxariScopeEstimator):
    """
    This estimator uses a bayesian version of linear regression. 
    """
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = linear_model.BayesianRidge()
        # TODO: Add polynomializer to estimation and optimization - Degreese 1-3
        # self._polynomializer = PolynomialFeatures()
        self._optimizer = optimizer or BayesianRegressorOptimizer()

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        degree = self.params.pop("degree", 1)
        X_ = self.polynomializer.set_params(degree=degree).fit_transform(X, y)
        indices = self.get_sample_indices(X_)
        self._estimator = self._estimator.set_params(**kwargs).fit(X_.iloc[indices], y.iloc[indices])
        return self

    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        return self._estimator.predict(X)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)

    def check_conformance(self):
        pass

    def get_config(self, deep=True):
        return {**self._estimator.get_params(), **super().get_config(deep)}