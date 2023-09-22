from base import OxariScopeEstimator, OxariOptimizer
import numpy as np
import pandas as pd
import optuna
from base.oxari_types import ArrayLike
from base.metrics import optuna_metric
from typing_extensions import Self
from sklearn.neural_network import MLPRegressor

class MLPOptimizer(OxariOptimizer):

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
            study_name=f"mlp_process_hp_tuning",
            direction="minimize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return study.best_params, df

    def score_trial(self, trial:optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
        param_space = {
            # "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [sizes]),
            "alpha": trial.suggest_float("alpha", 0.0001, 1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [400, 600, 800, 1000]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0001, 1, log=True),
            "max_iter": trial.suggest_categorical("max_iter", [2000]),
            "tol": trial.suggest_float("tol", 0.00001, 0.1, log=True),
            "early_stopping": trial.suggest_categorical("early_stopping", [True]),
           
        }
        hidden_layers = [trial.suggest_int(f"hidden_layers_{i}", 1, 5) for i in range(num_hidden_layers)]
        # hidden_layers_picked = [np.random.randint(1, 5) for i in range(num_hidden_layers)]
        # hidden_layers = trial.suggest_categorical("hidden_layer_sizes", [hidden_layers_picked])

        model = MLPRegressor(hidden_layer_sizes=hidden_layers, **param_space).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred)


class MLPEstimator(OxariScopeEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = MLPRegressor()
        self._optimizer = optimizer or MLPOptimizer()

    def fit(self, X, y, **kwargs) -> Self:
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self.params.pop("num_hidden_layers")
        # self.params.pop("hidden_layers")
        keys_to_remove = [key for key in self.params if key.startswith("hidden_layers_")]
        for key in keys_to_remove:
            self.params.pop(key)
        self._estimator = self._estimator.set_params(**self.params).fit(X, y)
        # self.coef_ = self._estimator.coef_
        return self
       
    def predict(self, X) -> ArrayLike:
        X = pd.DataFrame(X)
        return self._estimator.predict(X)

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        best_params = self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)
        return best_params

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)     

    def check_conformance(self):
        pass

    def get_config(self, deep=True):
        return {**self._estimator.get_params(), **super().get_config(deep)}
        
