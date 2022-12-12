from typing import Union, Tuple
from base import OxariScopeEstimator
from base import OxariOptimizer
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
import optuna
from pmdarima.metrics import smape
from sklearn import linear_model
from .linear.helper import PolynomialFeaturesMixin, NormalizedFeaturesMixin
from base.oxari_types import ArrayLike

DEBUG_NUM_TRIALS = True
NUM_TRIALS = 50 if not DEBUG_NUM_TRIALS else 10
NUM_STARTUP_TRIALS = 5 if not DEBUG_NUM_TRIALS else 1


class LROptimizer(PolynomialFeaturesMixin, OxariOptimizer):
    def __init__(self, num_trials=NUM_TRIALS, num_startup_trials=1, sampler=None, **kwargs) -> None:
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

    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        alpha = trial.suggest_float("alpha", 0.01, 5.0)
        l1_ratio = trial.suggest_float("l1_ratio", 0.01, 1.0)
        degree = trial.suggest_categorical("degree", list(range(1, 5)))
        X_train = self.polynomializer.set_params(degree=degree).fit_transform(X_train)
        X_val = self.polynomializer.set_params(degree=degree).fit_transform(X_val)

        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return smape(y_true=y_val, y_pred=y_pred)


class LinearRegressionEstimator(PolynomialFeaturesMixin, OxariScopeEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = linear_model.ElasticNet()
        # TODO: Add polynomializer to estimation and optimization - Degreese 1-3
        # self._polynomializer = PolynomialFeatures()
        self._optimizer = optimizer or LROptimizer()

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        degree = self.params.pop("degree", 1)
        X_, y_ = self.polynomializer.set_params(degree=degree).fit_transform(X, y)
        self._estimator = self._estimator.set_params(**kwargs).fit(X_, y_)
        return self

    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        X_ = self.polynomializer.transform(X)
        return self._estimator.predict(X_)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)

    def check_conformance(self):
        pass

    def get_config(self, deep=True):
        return {**self._estimator.get_params(), **super().get_config(deep)}


class GLMOptimizer(PolynomialFeaturesMixin, OxariOptimizer, NormalizedFeaturesMixin):
    def __init__(self, num_trials=20, num_startup_trials=NUM_STARTUP_TRIALS, sampler=None, **kwargs) -> None:
        super().__init__(
            num_trials=num_trials,
            num_startup_trials=num_startup_trials,
            sampler=sampler,
            **kwargs,
        )

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs) -> Tuple[dict, pd.DataFrame]:
        return super().optimize(X_train, y_train, X_val, y_val, **kwargs)

    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        alpha = trial.suggest_float("alpha", 0.01, 5.0)
        power = trial.suggest_categorical("power", (2, 3))
        degree = trial.suggest_categorical("degree", list(range(1, 5)))
        X_train = self.polynomializer.set_params(degree=degree).fit_transform(X_train, y_train)
        X_val = self.polynomializer.set_params(degree=degree).fit_transform(X_val, y_val)
        X_train = self.normalizer.fit_transform(X_train, y_train)
        X_val = self.normalizer.fit_transform(X_val, y_val)

        # TODO: Update scikit learn to the newest package. To change the solver of this estimator. solver='newton-cholesky'
        model = linear_model.TweedieRegressor(
            alpha=alpha,
            power=power,
        ).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return smape(y_true=y_val, y_pred=y_pred)


class GLMEstimator(PolynomialFeaturesMixin, NormalizedFeaturesMixin, OxariScopeEstimator):
    def __init__(self, degree=3, **kwargs) -> None:
        super().__init__(degree, **kwargs)
        self._estimator = linear_model.TweedieRegressor()
        self.set_optimizer(kwargs.pop('optimizer', GLMOptimizer()))

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        degree = self.params.pop("degree", 1)
        X_ = self.polynomializer.set_params(degree=degree).fit_transform(X, y)
        X_ = self.normalizer.fit_transform(X_, y)
        self._estimator = self._estimator.set_params(**kwargs).fit(X_, y)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        X_ = self.polynomializer.transform(X)
        X_ = self.normalizer.transform(X_)
        return self._estimator.predict(X_)