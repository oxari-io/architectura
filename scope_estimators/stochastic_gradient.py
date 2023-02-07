from typing import Union
from base import OxariScopeEstimator, OxariRegressor, OxariMixin, OxariOptimizer
import numpy as np
import pandas as pd
import optuna
from base.oxari_types import ArrayLike
from base.metrics import optuna_metric
import xgboost as xgb
import sklearn

from sklearn.linear_model import SGDRegressor
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.optimizers import SGD

class SGDOptimizer(OxariOptimizer):

    def __init__(self, base_learning_rate=0.01, policy='fixed', momentum=0.0, nesterov=1, sparse_dedup_aggregator=None, 
                 lars=None, **kwargs) -> None:
        super().__init__(
            base_learning_rate = base_learning_rate,
            policy = policy,
            momentum = momentum,
            nesterov = nesterov,
            sparse_dedup_aggregator = sparse_dedup_aggregator,
            lars = lars,
            init_kwargs = kwargs,
        )

    """ Alternatively, : """

    """ Parameters are set to default for now but:
        The penalty/regularization term 'l2' is the standard regularizer for linear SVM models.
        alpha: The higher the value, the stronger the regularization
        l1_ratio: only used if 'penalty' is "elasticnet".
        fit_intercept: If False, the data is assumed to be already centered.
        epsilon: Epsilon in the epsilon-insensitive loss functions
        validation_fraction and n_iter_no_change and tol: used only if EarlyStopping = True
    """
    # def __init__(self, loss='squared_error', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, 
    # tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, 
    # early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False, **kwargs) -> None:
    #     super().__init__(
    #         loss=loss,
    #         penalty=penalty,
    #         alpha=alpha,
    #         l1_ratio=l1_ratio,
    #         fit_intercept=fit_intercept,
    #         max_iter=max_iter,
    #         tol=tol,
    #         shuffle=shuffle,
    #         verbose=verbose,
    #         epsilon=epsilon,
    #         random_state=random_state,
    #         learning_rate=learning_rate,
    #         eta0=eta0,
    #         power_t=power_t,
    #         early_stopping=early_stopping,
    #         validation_fraction=validation_fraction,
    #         n_iter_no_change=n_iter_no_change,
    #         warm_start=warm_start,
    #         average=average,
    #         init_kwargs = kwargs,
    #     )

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
            study_name=f"sgd_process_hp_tuning",
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
        alpha = trial.suggest_float("alpha", 0.0001, 0.1)
        
            
        max_size = len(X_train)
        sample_size = int(max_size*0.1)
        indices = np.random.randint(0, max_size, sample_size)
        model = SGDRegressor(epsilon=epsilon, alpha=alpha).fit(X_train.iloc[indices], y_train.iloc[indices])
        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred)




class SGDEstimator(OxariScopeEstimator):
    def __init__(self, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self._estimator = SGDRegressor()
        self._optimizer = optimizer or SGDOptimizer()

    def fit(self, X, y, **kwargs) -> "SGDEstimator":
        max_size = len(X)
        sample_size = int(max_size*0.1)
        indices = np.random.randint(0, max_size, sample_size)   
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self._estimator = self._estimator.set_params(**self.params).fit(X.iloc[indices], y.iloc[indices].values.ravel())
        # self.coef_ = self._estimator.coef_
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
        
       

    # # we can add more parameters if we want
    # def fit(self, X, y, **kwargs) -> "OxariRegressor":
    #     return self.fit(X, y)
    
    # def predict(self, X:ArrayLike, **kwargs) -> ArrayLike:
    #     return self.predict(X)

    # # alternative to "def _set_meta"
    # def set_params(self, X:ArrayLike, **kwargs) -> ArrayLike:
    #     self.feature_names_in_ = list(X.columns)
    #     self.n_features_in_ = len(self.feature_names_in_)