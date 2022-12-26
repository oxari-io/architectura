from typing import Union, Tuple
from base import OxariScopeEstimator
from base import OxariOptimizer, OxariTransformer
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
import optuna
from sklearn import linear_model
from .linear.helper import PolynomialFeaturesMixin, NormalizedFeaturesMixin
from base.oxari_types import ArrayLike
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from base.metrics import optuna_metric

DEBUG_NUM_TRIALS = True
NUM_TRIALS = 50 if not DEBUG_NUM_TRIALS else 10
NUM_STARTUP_TRIALS = 5 if not DEBUG_NUM_TRIALS else 1


class LROptimizer(OxariOptimizer):
    def __init__(self, n_trials=NUM_TRIALS, n_startup_trials=1, sampler=None, **kwargs) -> None:
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
            study_name=f"{self.__class__.__name__}_process_hp_tuning",
            direction="minimize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations

        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return study.best_params, df

    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        alpha = trial.suggest_float("alpha", 0.1, 5.0)
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 1.0)
        degree = trial.suggest_categorical("degree", list(range(1, 10)))

        preprocessor = LinearRegressionEstimator._make_model_specific_preprocessor(X_train, y_train, degree=degree)
        X_train = preprocessor.transform(X_train)
        X_val = preprocessor.transform(X_val)

        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred)


class LinearRegressionEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator = linear_model.ElasticNet()
        self.set_optimizer(kwargs.pop("optimizer", LROptimizer()))

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        degree = self.params.pop("degree", 1)
        self._sub_preprocessor = LinearRegressionEstimator._make_model_specific_preprocessor(X, y, degree=degree)
        X_ = self._sub_preprocessor.transform(X)
        self._estimator = self._estimator.set_params(**self.params).fit(X_, y)
        return self

    @staticmethod
    def _make_model_specific_preprocessor(X, y, **kwargs) -> OxariTransformer:
        return Pipeline([
            ('polinomial', PolynomialFeatures(degree=kwargs.pop("degree"), include_bias=False)),
        ]).fit(X, y, **kwargs)

    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        X_ = self._sub_preprocessor.transform(X)
        return self._estimator.predict(X_)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)

    def check_conformance(self):
        pass

    def get_config(self, deep=True):
        return {**self._estimator.get_params(), **super().get_config(deep)}


class GLMOptimizer(OxariOptimizer):
    def __init__(self, n_trials=50, n_startup_trials=NUM_STARTUP_TRIALS, sampler=None, **kwargs) -> None:
        super().__init__(
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
            sampler=sampler,
            **kwargs,
        )

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs) -> Tuple[dict, pd.DataFrame]:
        return super().optimize(X_train, y_train, X_val, y_val, **kwargs)

    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        power = trial.suggest_categorical("power", (0, 2, 3))
        degree = trial.suggest_categorical("degree", list(range(1, 5)))

        preprocessor = GLMEstimator._make_model_specific_preprocessor(X_train, y_train, degree=degree)
        X_train = preprocessor.transform(X_train)
        X_val = preprocessor.transform(X_val)

        # TODO: Update scikit learn to the newest package. To change the solver of this estimator. solver='newton-cholesky'
        model = linear_model.TweedieRegressor(
            alpha=alpha,
            power=power,
            max_iter=500,
        ).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred)


class GLMEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._estimator = linear_model.TweedieRegressor()
        self.set_optimizer(kwargs.pop('optimizer', GLMOptimizer()))

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        degree = self.params.pop("degree", 1)
        self._sub_preprocessor = GLMEstimator._make_model_specific_preprocessor(X, y, degree=degree)
        X_ = self._sub_preprocessor.transform(X)
        self._estimator = self._estimator.set_params(**kwargs, max_iter=500).fit(X_, y)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        X_ = self._sub_preprocessor.transform(X)
        return self._estimator.predict(X_)

    @staticmethod
    def _make_model_specific_preprocessor(X, y, **kwargs) -> OxariTransformer:
        return Pipeline([
            ('polinomial', PolynomialFeatures(degree=kwargs.pop("degree"), include_bias=False)),
            ('minmax', MinMaxScaler()),
        ]).fit(X, y, **kwargs)