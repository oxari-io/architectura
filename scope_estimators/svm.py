from typing import Union
from base import OxariScopeEstimator
from base.common import OxariOptimizer
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import optuna
from pmdarima.metrics import smape
from base.oxari_types import ArrayLike


class SVROptimizer(OxariOptimizer):
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
            study_name=f"svm_process_hp_tuning",
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
        epsilon = trial.suggest_float("epsilon", 0.01, 0.2)
        C = trial.suggest_float("C", 0.01, 2.0)
        
            
        max_size = len(X_train)
        sample_size = int(max_size*0.1)
        indices = np.random.randint(0, max_size, sample_size)
        model = SVR(epsilon=epsilon, C=C).fit(X_train.iloc[indices], y_train.iloc[indices])
        y_pred = model.predict(X_val)

        return smape(y_true=y_val, y_pred=y_pred)


class SupportVectorEstimator(OxariScopeEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = SVR()
        self._optimizer = optimizer or SVROptimizer()

    def fit(self, X, y, **kwargs) -> "SupportVectorEstimator":
        max_size = len(X)
        sample_size = int(max_size*0.1)
        indices = np.random.randint(0, max_size, sample_size)        
        self._estimator = self._estimator.set_params(**self.params).fit(X.iloc[indices], y.iloc[indices])
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