from __future__ import annotations
from typing import Union

from base import OxariScopeEstimator, DefaultRegressorEvaluator, DefaultOptimizer, OxariOptimizer, OxariTransformer
from base.helper import BucketScopeDiscretizer
import numpy as np
import pandas as pd
from scope_estimators.mma.classifier import BucketClassifier, ClassifierOptimizer, BucketClassifierEvauator
from scope_estimators.mma.regressor import BucketRegressor, RegressorOptimizer
from base.oxari_types import ArrayLike
from sklearn.linear_model import RidgeCV, Ridge
import optuna
from base.metrics import optuna_metric
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
N_TRIALS = 1
N_STARTUP_TRIALS = 1

Model = RidgeCV

class LRAOptimizer(OxariOptimizer):
    def __init__(self, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS, sampler=None, **kwargs) -> None:
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
        _estimators = {}
        degree = trial.suggest_categorical("degree", list(range(1, 3)))

        preprocessor = LinearRegressionArmyEstimator._make_model_specific_preprocessor(X_train, y_train, degree=degree)
        X_train = preprocessor.transform(X_train)
        X_val = preprocessor.transform(X_val)
        results = np.zeros((2, X_val.shape[1], X_val.shape[0]))
        for idx in range(X_train.shape[1]):
            x_train = np.array(X_train)[:, idx, None]
            x_val = np.array(X_val)[:, idx, None]
            _estimators[idx] = Model().fit(x_train, y_train)
            results[0, idx] = _estimators[idx].predict(x_val)
            results[1, idx] = -_estimators[idx].best_score_
        results[1] = 1/results[1]
        results[1] = (results[1] / results[1].sum(axis=0))
        weighted_preds = results[0] * results[1]
        y_pred = weighted_preds.sum(axis=0)

        return optuna_metric(y_true=y_val, y_pred=y_pred)


class LinearRegressionArmyEstimator(OxariScopeEstimator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimators = {}
        self.set_evaluator(DefaultRegressorEvaluator())
        self.set_optimizer(LRAOptimizer())

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        degree = self.params.pop("degree")
        preprocessor = LinearRegressionArmyEstimator._make_model_specific_preprocessor(X, y, degree=degree)
        X = preprocessor.transform(X)
        for idx in range(X.shape[1]):
            x = np.array(X)[:, idx, None]
            self._estimators[idx] = Model(**self.params).fit(x, y)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        results = np.zeros((2, X.shape[1], X.shape[0]))
        for idx in range(X.shape[1]):
            x = np.array(X)[:, idx, None]
            results[0, idx] = self._estimators[idx].predict(x)
            results[1, idx] = -self._estimators[idx].best_score_
        results[1] = 1/results[1]
        results[1] = (results[1] / results[1].sum(axis=0))
        weighted_preds = results[0] * results[1]
        y_pred = weighted_preds.sum(axis=0)
        return y_pred


    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)
    
    @staticmethod
    def _make_model_specific_preprocessor(X, y, **kwargs) -> OxariTransformer:
        return Pipeline([
            ('polinomial', PolynomialFeatures(degree=kwargs.pop("degree"), include_bias=False)),
        ]).fit(X, y, **kwargs)
