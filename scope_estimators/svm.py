from sklearn.preprocessing import PolynomialFeatures
from typing_extensions import Self
import numpy as np
import optuna
import pandas as pd
from sklearn.svm import SVR, LinearSVR

from base import OxariScopeEstimator
from base.common import OxariOptimizer, OxariTransformer
from base.metrics import optuna_metric
from base.oxari_types import ArrayLike
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline, make_pipeline

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
        study:optuna.Study = optuna.create_study(
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

        return optuna_metric(y_true=y_val, y_pred=y_pred)

class LinearSVROptimizer(OxariOptimizer):

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
            study_name=f"linear_svr_process_hp_tuning",
            direction="minimize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return study.best_params, df

    def score_trial(self, trial:optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        
        param_space = {
            "epsilon": trial.suggest_float("epsilon", 1.0, 20.24), # depends on scale of target value, range: [0, 20.24 (y_train_max)]
            "C": trial.suggest_float("C", 0.1, 1000, log=True),
            "loss": trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
            "intercept_scaling": trial.suggest_float("intercept_scaling", 1.0, 150.0, step=0.5),
            "max_iter": trial.suggest_categorical("max_iter", [7000])
        }
        
        degree = trial.suggest_int("degree", 1, 1)

        preprocessor = LinearSVREstimator._make_model_specific_preprocessor(X_train, y_train, degree=degree)
        X_train = preprocessor.transform(X_train)
        X_val = preprocessor.transform(X_val)        
        model = LinearSVR(**param_space).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred)



class SupportVectorEstimator(OxariScopeEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = SVR()
        self._optimizer = optimizer or SVROptimizer(**kwargs)
        self.logger.warning(f"Better use {FastSupportVectorEstimator}! It's much faster and performs better.")

    def fit(self, X, y, **kwargs) -> Self:
        self.logger.warning(f"AGAIN use {FastSupportVectorEstimator}! It's much faster and performs better.")
        max_size = len(X)
        sample_size = int(max_size*0.1)
        indices = np.random.randint(0, max_size, sample_size)   
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self._estimator = self._estimator.set_params(**self.params).fit(X.iloc[indices], y.iloc[indices].values.ravel())
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

    

class FastSVROptimizer(SVROptimizer):
    def score_trial(self, trial: optuna.Trial, X_train, y_train, X_val, y_val, **kwargs):
        epsilon = trial.suggest_float("epsilon", 0.01, 0.2)
        C = trial.suggest_float("C", 0.01, 2.0)
        model = make_pipeline(Nystroem(), LinearSVR(epsilon=epsilon, C=C)).fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred)
        
class FastSupportVectorEstimator(SupportVectorEstimator):
    def __init__(self, optimizer:OxariOptimizer|None =None, **kwargs):
        super().__init__(optimizer, **kwargs)
        self._estimator = make_pipeline(Nystroem(), LinearSVR())
        self._optimizer = optimizer or FastSVROptimizer()  

    def fit(self, X, y, **kwargs) -> Self:
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self._estimator.named_steps["linearsvr"].set_params(**self.params)
        self._estimator = self._estimator.fit(X, y.values.ravel())
        return self
    



class LinearSVREstimator(OxariScopeEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = LinearSVR()
        self._optimizer = optimizer or LinearSVROptimizer()

    def fit(self, X, y, **kwargs) -> Self:
        degree = self.params.pop("degree", 1)
        self._sub_preprocessor = LinearSVREstimator._make_model_specific_preprocessor(X, y, degree=degree)
        X_ = self._sub_preprocessor.transform(X)
        max_size = len(X)
        sample_size = int(max_size*0.1)
        indices = np.random.randint(0, max_size, sample_size)
        X_ = pd.DataFrame(X_)
        y = pd.DataFrame(y)
        self._estimator = self._estimator.set_params(**self.params).fit(X_.iloc[indices], y.iloc[indices].values.ravel())
        return self
    
    @staticmethod
    def _make_model_specific_preprocessor(X, y, **kwargs) -> OxariTransformer:
        return Pipeline([
            ('polinomial', PolynomialFeatures(degree=kwargs.pop("degree"), include_bias=False)),
        ]).fit(X, y, **kwargs)
       
    def predict(self, X) -> ArrayLike:
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
        
