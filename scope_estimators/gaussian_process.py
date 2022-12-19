from typing import Union
from base import OxariScopeEstimator, ReducedDataMixin
from base.common import OxariOptimizer
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
import optuna
from pmdarima.metrics import smape

STANDARD_KERNEL = kernels.DotProduct() + kernels.WhiteKernel()


class GPOptimizer(ReducedDataMixin, OxariOptimizer):
    def __init__(self, num_trials=20, num_startup_trials=5, sampler=None, **kwargs) -> None:
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
            study_name=f"gaussian_process_hp_tuning",
            direction="minimize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.num_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return study.best_params, df

    # TODO: Find better optimization ranges for the GaussianProcessEstimator
    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        outer_alpha = trial.suggest_float("outer_alpha", 1e-8, 1, log=True)
        length_scale = trial.suggest_float("length_scale", 1e-8, 1, log=True)
        alpha = trial.suggest_float("alpha", 1e-8, 1, log=True)
        sigma = trial.suggest_float("sigma", 1e-8, 1, log=True)
        nu = trial.suggest_categorical("nu", [0.5, 1.5, 2.5, np.inf])
        noise = trial.suggest_float("noise", 1e-8, 1, log=True)
        main_kernel = trial.suggest_categorical("main_kernel", ["rbf", "rq", "dot", "matern"])

        kernel = GaussianProcessEstimator._compose_kernel(length_scale, alpha, sigma, nu, noise, main_kernel)

        indices = self.get_sample_indices(X_train)
        model = GaussianProcessRegressor(kernel=kernel, alpha=outer_alpha, normalize_y=True).fit(X_train.iloc[indices], y_train.iloc[indices])
        y_pred = model.predict(X_val)

        return smape(y_true=y_val, y_pred=y_pred)

    # TODO: Explore kernel setups for the GP that have a better fit with the data. https://www.cs.toronto.edu/~duvenaud/cookbook/



class GaussianProcessEstimator(ReducedDataMixin, OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator = GaussianProcessRegressor()
        self.set_optimizer(kwargs.pop("optimizer",  GPOptimizer()))

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        outer_alpha = self.params.pop('outer_alpha')
        kernel = GaussianProcessEstimator._compose_kernel(**self.params)
        indices = self.get_sample_indices(X)
        self._estimator = self._estimator.set_params(alpha=outer_alpha, kernel=kernel, normalize_y=True).fit(X.iloc[indices], y.iloc[indices])
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
    
    @staticmethod
    def _compose_kernel(length_scale, alpha, sigma, nu, noise, main_kernel):
        kernel_noise = kernels.WhiteKernel(noise_level=noise)
        kernel_noise = kernels.ConstantKernel()
        kernel = kernel_noise
        if main_kernel == "rbf":
            kernel += kernels.RBF(length_scale=length_scale)
        if main_kernel == "rq":
            kernel += kernels.RationalQuadratic(length_scale=length_scale, alpha=alpha)
        if main_kernel == "dot":
            kernel += kernels.DotProduct(sigma_0=sigma)
        if main_kernel == "matern":
            kernel += kernels.Matern(length_scale=length_scale, nu=nu)
        return kernel