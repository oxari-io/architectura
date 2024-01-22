from collections import defaultdict
from pathlib import Path
from typing import Dict

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from pmdarima.metrics import smape
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor
from sklearn.metrics import median_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor, VotingRegressor)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize
from tqdm import tqdm

from base.common import OxariOptimizer, OxariRegressor
from base.metrics import cv_metric, optuna_metric
from scope_estimators.stochastic_gradient import SGDEstimator

# from sklearn.metrics import root_mean_squared_error as rmse
# from sklearn.metrics import mean_absolute_percentage_error as mape
# from model.misc.metrics import mape

# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# from ngboost import NGBRegressor
# from scipy import stats
# from pprint import pprint


# from model.misc.mappings import NumMapping as Mapping
# from model.misc.metrics import adjusted_r_squared
# from model.misc.hyperparams_tuning import tune_hps_regressors

# from model.abstract_base_class import MLModelInterface

# from model.misc.ML_toolkit import add_bucket_label, check_scope

OBJECT_DIR = Path("model/objects")
METRICS_DIR = Path("model/metrics")
DATA_DIR = Path("model/data")
OPTUNA_DIR = Path("model/optuna")


class RegressorOptimizer(OxariOptimizer):
    def __init__(self, n_trials=2, n_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.sampler = sampler or optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials, warn_independent_sampling=False)
        self.scope = None
        self.models = [
            # ("GBR", GradientBoostingRegressor), # Is fundamentally the same as XGBOOST but XGBoost is better - https://stats.stackexchange.com/a/282814/361976
            ("RFR", RandomForestRegressor),
            # ("XGB", xgb.XGBRegressor),
            ("LGB", lgb.LGBMRegressor),
            ("KNN", KNeighborsRegressor),
            ("EXF", ExtraTreesRegressor),
        ]

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Explores the hyperparameter search space with optuna.
        Creates csv and pickle files with the saved hyperparameters for regression

        Parameters:
        name (string): name of the model RFR --> RandomForestRegressor; GBR --> GradientBoostingRegressor; XGB --> XGBoostRegressor
        X_train (numpy array): training data 
        y_train (numpy array): training data 
        X_val (numpy array): validation data
        y_val (numpy array): validation data 
        num_startup_trials (int): 
        n_trials (int): 

        Return:
        study.best_params (data structure): contains the best found hyperparameters within the given space
        """
        # selecting the models that will be trained to build the voting regressor --> tuple(name, model)
        # TODO: Use ExtraTree instead of RandomForest as it is much faster with similar performance
        
        # scores will be used to compute weights for voting mechanism
        info = defaultdict(lambda: defaultdict(dict))
        # candidates will be given to VotingRegressor
        candidates = defaultdict(lambda: defaultdict(dict))
        grp_train = kwargs.get('grp_train')
        grp_val = kwargs.get('grp_val')
        for n in np.unique(grp_train).astype(np.int):
            selector_train = (grp_train == n)[:, 0]
            selector_val = (grp_val == n)[:, 0]
            
            # candidates[n] = defaultdict(dict)
            # info[n] = defaultdict(dict)
            # bucket_name = f"bucket_{n}"
            bucket_name = n
            for name, Model in self.models:

                # print(f"Training {name} ... ")
                self.logger.debug(f"Traning {name}")

                study = optuna.create_study(
                    study_name=f"regressor_{name}_hp_tuning",
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials, warn_independent_sampling=False),
                )
                study.optimize(
                    lambda trial: self.score_trial(trial, name, X_train[selector_train], y_train[selector_train].values, X_val[selector_val], y_val[selector_val].values),
                    n_trials=self.n_trials,
                    show_progress_bar=False)

                candidates[bucket_name][name]["best_params"] = study.best_params
                candidates[bucket_name][name]["Model"] = Model
                info[bucket_name][name] = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return {"candidates": candidates}, info

    def score_trial(self, trial: optuna.Trial, regr_name: str, X_train, y_train, X_val, y_val):
        # TODO: add docstring here
        # TODO: Try MSLE as optimization objective https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        if regr_name == "GBR":

            param_space = {
                # The number of boosting stages to perform
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 21, 3),
                # The fraction of samples to be used for fitting the individual base learners
                "subsample": trial.suggest_float("subsample", 0.5, 0.9, step=0.1),
                # The number of features to consider when looking for the best split
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt"]),
            }

            model = GradientBoostingRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "RFR":
            #  definin search space of RandomForest
            param_space = {
                # this parameter means using the GPU when training our model to speedup the training process
                # 'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                'n_estimators': trial.suggest_int("n_estimators", 100, 500, 100),
                'max_features': trial.suggest_float('max_features', 0.2, 0.8, step=0.2),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 2, 20, 2),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'poisson']),
                # The number of features to consider when looking for the best split
                # 'min_samples_split': trial.suggest_int("min_samples_split", 2, 12, 2),
                "n_jobs": trial.suggest_categorical('n_jobs', [-1])
            }

            model = RandomForestRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "KNN":
            #  definin search space of KNN
            param_space = {
                'n_neighbors': trial.suggest_int("n_neighbors", 3, 11, 2),
                'p': trial.suggest_categorical('p', [1,2]),
                "n_jobs": trial.suggest_categorical('n_jobs', [-1])
            }

            model = KNeighborsRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "EXF":
            #  definin search space of KNN
            param_space = {
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, 100),
                'max_features': trial.suggest_float('max_features', 0.2, 0.8, step=0.2),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 2, 20, 2),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'poisson']),
                # The number of features to consider when looking for the best split
                # 'min_samples_split': trial.suggest_int("min_samples_split", 2, 12, 2),
                "n_jobs": trial.suggest_categorical('n_jobs', [-1])
            }

            model = ExtraTreesRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "XGB":

            param_space = {
                # L2 regularization term on weights.
                # 'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                # # L1 regularization term on weights.
                # 'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                # Subsample ratio of columns when constructing each tree.
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.7, step=0.1),
                # Subsample ratio of the training instances.
                'subsample': trial.suggest_float('subsample', 0.4, 0.7, step=0.1),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, 100),

                # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),

                # 'random_state': trial.suggest_categorical('random_state', [2020]),
                # If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5, 1),
            }

            model = xgb.XGBRegressor(**param_space)
            model.fit(X_train, y_train)
        if regr_name == "LGB":

            param_space = {
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9, step=0.1),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 5, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9, step=0.1),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, 100),
            }

            model = lgb.LGBMRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "DT":

            param_space = {
                # this parameter means using the GPU when training our model to speedup the training process
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                # 'n_estimators': trial.suggest_int("n_estimators", 100, 500, 100),
                'max_features': trial.suggest_float('max_features', 0.1, 0.9, step=0.1),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20, step=2),
                # 'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'poisson']),
                # The number of features to consider when looking for the best split
                # 'min_samples_split': trial.suggest_int("min_samples_split", 2, 12, 2),
                # "n_jobs": trial.suggest_categorical('n_jobs', [-1])
            }

            model = DecisionTreeRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "KR":

            param_space = {
                'alpha': trial.suggest_float('alpha', 0.1, 1, step=0.1),
            }

            model = make_pipeline(Nystroem(), Ridge(**param_space))
            model.fit(X_train, y_train)

        if regr_name == "KSGD":

            param_space = {
                'penalty': 'elasticnet',
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 1, step=0.1),
                'learning_rate': 'optimal',
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            }

            model = make_pipeline(Nystroem(), SGDRegressor(**param_space))
            model.fit(X_train, y_train)

        if regr_name == "KSVR":

            param_space = {
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
                'gamma': trial.suggest_float('gamma', 2e-15, 2e3, log=True),
            }

            model = make_pipeline(Nystroem(), SVR(**param_space))
            model.fit(X_train, y_train)

        # if regr_name == "ADB":
        #     param_space = {
        #         "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=200),
        #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        #         'splitter': trial.suggest_categorical('max_depth', ["best", "random"]),
        #         'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
        #     }

        #     model = AdaBoostRegressor(**param_space)
        #     model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        return self.metric(y_val, y_pred)

# Optimizer used in 'experiment_optuna_metric'
class ExperimentOptimizer(RegressorOptimizer):
    def __init__(self, metric, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
    
    def score_trial(self, trial: optuna.Trial, regr_name: str, X_train, y_train, X_val, y_val):
        # TODO: add docstring here
        # TODO: Try MSLE as optimization objective https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        if regr_name == "GBR":

            param_space = {
                # The number of boosting stages to perform
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 21, 3),
                # The fraction of samples to be used for fitting the individual base learners
                "subsample": trial.suggest_float("subsample", 0.5, 0.9, step=0.1),
                # The number of features to consider when looking for the best split
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt"]),
            }

            model = GradientBoostingRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "RFR":
            #  definin search space of RandomForest
            param_space = {
                # this parameter means using the GPU when training our model to speedup the training process
                # 'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                'n_estimators': trial.suggest_int("n_estimators", 100, 500, 100),
                'max_features': trial.suggest_float('max_features', 0.2, 0.8, step=0.2),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 2, 20, 2),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'poisson']),
                # The number of features to consider when looking for the best split
                # 'min_samples_split': trial.suggest_int("min_samples_split", 2, 12, 2),
                "n_jobs": trial.suggest_categorical('n_jobs', [-1])
            }

            model = RandomForestRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "KNN":
            #  definin search space of KNN
            param_space = {
                'n_neighbors': trial.suggest_int("n_neighbors", 3, 11, 2),
                'p': trial.suggest_categorical('p', [1,2]),
                "n_jobs": trial.suggest_categorical('n_jobs', [-1])
            }

            model = KNeighborsRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "EXF":
            #  definin search space of KNN
            param_space = {
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, 100),
                'max_features': trial.suggest_float('max_features', 0.2, 0.8, step=0.2),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 2, 20, 2),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'poisson']),
                # The number of features to consider when looking for the best split
                # 'min_samples_split': trial.suggest_int("min_samples_split", 2, 12, 2),
                "n_jobs": trial.suggest_categorical('n_jobs', [-1])
            }

            model = ExtraTreesRegressor(**param_space)
            model.fit(X_train, y_train)

        if regr_name == "XGB":

            param_space = {
                # L2 regularization term on weights.
                # 'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                # # L1 regularization term on weights.
                # 'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                # Subsample ratio of columns when constructing each tree.
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.7, step=0.1),
                # Subsample ratio of the training instances.
                'subsample': trial.suggest_float('subsample', 0.4, 0.7, step=0.1),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, 100),

                # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),

                # 'random_state': trial.suggest_categorical('random_state', [2020]),
                # If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5, 1),
            }

            model = xgb.XGBRegressor(**param_space)
            model.fit(X_train, y_train)
        if regr_name == "LGB":

            param_space = {
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9, step=0.1),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9, step=0.1),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int("n_estimators", 100, 300, 100),
            }

            model = lgb.LGBMRegressor(**param_space)
            model.fit(X_train, y_train)

        # if regr_name == "ADB":
        #     param_space = {
        #         "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=200),
        #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        #         'splitter': trial.suggest_categorical('max_depth', ["best", "random"]),
        #         'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
        #     }

        #     model = AdaBoostRegressor(**param_space)
        #     model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        return optuna_metric(y_true=y_val, y_pred=y_pred, metric=self.metric)
    
class StackingRegressorOptimizer(RegressorOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = [
            ('DT', DecisionTreeRegressor),
            ('KNN', KNeighborsRegressor),
            ('KR', (Nystroem, Ridge)),
            ('KSGD', (Nystroem, SGDRegressor)),
        ]
        # self.models_level_2 = [
        #     ('EXF', ExtraTreesRegressor),
        #     ('KSVR', (Nystroem, SVR)),
        #     ('RFR', RandomForestRegressor),
        # ]

class BucketRegressor(OxariRegressor):
    # TODO: add docstring
    def __init__(self, **kwargs):
        # self.scope = check_scope(scope)
        self.voting_regressors_: Dict[int, VotingRegressor] = {}
        self.bucket_specifics_ = {"scores":{}, "weights":{}}
        # self.bucket_specifics_ = {}

    def fit(self, X, y, **kwargs):
        """
        The main training loop that calls all the other functions
        Subsets data, splits in training, test and validation, builds and trains voting regressor, computes error metrics

        """
        groups = kwargs.get('groups')
        # TODO: Doesn't work with CVPipeline as candidates are not set. Needs a fix.
        regressor_kwargs = dict(self.params.get("candidates"))
        trained_candidates = {}
        total = len(regressor_kwargs) * len(list(regressor_kwargs.items())[0][1])
        pbar = tqdm(desc="MMA-Regressor", total=total)
        
        for bucket, candidates_data in regressor_kwargs.items():
            bucket_name = f"bucket_{bucket}"
            selector = groups == bucket
            is_any = np.sum(selector) > 10
            X_subset = X[selector] if is_any else X
            y_subset = y[selector] if is_any else y
            # X_train, X_val, y_train, y_val = train_test_split(X[selector], y[selector], test_size=0.3) if is_any else train_test_split(X, y, test_size=0.3)
            for name, candidate_data in candidates_data.items():
                best_params = candidate_data.get("best_params")
                ModelConstructor = candidate_data.get("Model")
                
                model = ModelConstructor(**best_params)

                # calculate the score of each individual model to weight voting mechanism
                # NOTE: Could also be done faster by not using cv
                trained_candidates = self._compute_model_score(X_subset, y_subset, trained_candidates, name, candidate_data, model)
                pbar.update(1)
            v_regressor = self._construct_voting_regressor(X_subset, y_subset, trained_candidates)
            self.voting_regressors_[bucket] = v_regressor
            y_hat = self.voting_regressors_[bucket].predict(X_subset)
            y_true = y_subset
            self.bucket_specifics_["scores"][bucket_name] = self.evaluate(y_true, y_hat)
            self.bucket_specifics_["weights"][bucket_name] = {e[0]: weight for e, weight in zip(v_regressor.estimators, v_regressor.weights)}

        return self

    def _cv_metric(self, estimator, X, y):
        y_hat = estimator.predict(X)
        return smape(y, y_hat)

    def _compute_model_score(self, X, y, trained_candidates, name, candidate_data, model):
        model_score = np.mean(cross_val_score(model, X, y, scoring=self._cv_metric)) 
        trained_candidates[name] = {"model": model, "score": 1/(model_score + np.finfo(np.float64).eps), **candidate_data}
        return trained_candidates

    def _construct_voting_regressor(self, X, y, trained_candidates):
        weights = np.array([v["score"] for _, v in trained_candidates.items()])
        normalised_weights = weights / np.sum(weights)
        models = [(name, v["model"]) for name, v in trained_candidates.items()]
        v_regressor = VotingRegressor(estimators=models, weights=normalised_weights, n_jobs=-1).fit(X, y)
        return v_regressor

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        best_params, info = self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)
        return best_params, info

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred)

    def predict(self, X: pd.DataFrame, **kwargs):
        """
        Voting regressor computes prediction

        Parameters:
        data (pandas.DataFrame): pre-processed dataset

        Return:
        predictions (numpy array): the predicted values
        """
        groups = kwargs.get('groups')
        y_pred = np.zeros(X.shape[0])

        for bucket, voting_regressor in self.voting_regressors_.items():
            selector = (bucket == groups).flatten()
            if not np.any(selector):
                continue
            y_pred[selector] = voting_regressor.predict(X[selector])

        return y_pred

    def set_params(self, **params):
        self.params = params
        return self

    def get_config(self, deep=True):
        return self.params


class EvenWeightBucketRegressor(BucketRegressor):
    def _construct_voting_regressor(self, X, y, trained_candidates):
        weights = np.array([1 for _, v in trained_candidates.items()])
        normalised_weights = weights / np.sum(weights)
        models = [(name, v["model"]) for name, v in trained_candidates.items()]
        v_regressor = VotingRegressor(estimators=models, weights=normalised_weights, n_jobs=-1).fit(X, y)
        return v_regressor

class AlternativeCVMetricBucketRegressor(BucketRegressor):
    def _compute_model_score(self, X, y, trained_candidates, name, candidate_data, model):
        model_score = np.mean(cross_val_score(model, X, y, scoring=self._cv_metric)) 
        trained_candidates[name] = {"model": model, "score": 1/(model_score + np.finfo(np.float64).eps), **candidate_data}
        return trained_candidates
    
    def _cv_metric(self, estimator, X, y):
        y_hat = estimator.predict(X)
        return median_absolute_error(y, y_hat)

class CombinedBucketRegressor(EvenWeightBucketRegressor, AlternativeCVMetricBucketRegressor):
    pass  

class BucketStackingRegressor(OxariRegressor):
    # TODO: add docstring
    def __init__(self, **kwargs):
        # self.scope = check_scope(scope)
        self.stacking_regressors_: Dict[int, StackingRegressor] = {}
        self.bucket_specifics_ = {"scores":{}}
        # self.bucket_specifics_ = {}

    def fit(self, X, y, **kwargs):
        """
        The main training loop that calls all the other functions
        Subsets data, splits in training, test and validation, builds and trains stacking regressor, computes error metrics

        """
        groups = kwargs.get('groups')
        # TODO: Doesn't work with CVPipeline as candidates are not set. Needs a fix.
        regressor_kwargs = dict(self.params.get("candidates"))
        trained_candidates = {}
        total = len(regressor_kwargs) * len(list(regressor_kwargs.items())[0][1])
        pbar = tqdm(desc="MMA-Regressor", total=total)
        
        for bucket, candidates_data in regressor_kwargs.items():
            bucket_name = f"bucket_{bucket}"
            selector = groups == bucket
            is_any = np.sum(selector) > 10
            X_subset = X[selector] if is_any else X
            y_subset = y[selector] if is_any else y
            # X_train, X_val, y_train, y_val = train_test_split(X[selector], y[selector], test_size=0.3) if is_any else train_test_split(X, y, test_size=0.3)
            for name, candidate_data in candidates_data.items():
                best_params = candidate_data.get("best_params")
                ModelConstructor = candidate_data.get("Model")
                
                if isinstance(ModelConstructor, tuple):
                    model = make_pipeline(ModelConstructor[0](), ModelConstructor[1](**best_params))
                else:
                    model = ModelConstructor(**best_params)

                # calculate the score of each individual model to weight stacking mechanism
                # NOTE: Could also be done faster by not using cv
                trained_candidates = self._compute_model_score(X_subset, y_subset, trained_candidates, name, candidate_data, model)
                pbar.update(1)
            s_regressor = self._construct_stacking_regressor(X_subset, y_subset, trained_candidates)
            self.stacking_regressors_[bucket] = s_regressor
            y_hat = self.stacking_regressors_[bucket].predict(X_subset)
            y_true = y_subset
            self.bucket_specifics_["scores"][bucket_name] = self.evaluate(y_true, y_hat)

        return self

    def _cv_metric(self, estimator, X, y):
        y_hat = estimator.predict(X)
        return smape(y, y_hat)

    def _compute_model_score(self, X, y, trained_candidates, name, candidate_data, model):
        model_score = np.mean(cross_val_score(model, X, y, scoring=self._cv_metric)) 
        trained_candidates[name] = {"model": model, "score": 1/(model_score + np.finfo(np.float64).eps), **candidate_data}
        return trained_candidates

    def _construct_stacking_regressor(self, X, y, trained_candidates):
        models = [(name, v["model"]) for name, v in trained_candidates.items()]
        final_estimator = GradientBoostingRegressor(n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1, random_state=42)
        s_regressor = StackingRegressor(estimators=models, final_estimator=final_estimator, n_jobs=-1).fit(X, y)
        return s_regressor

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        best_params, info = self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)
        return best_params, info

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred)

    def predict(self, X: pd.DataFrame, **kwargs):
        """
        Stacking regressor computes prediction

        Parameters:
        data (pandas.DataFrame): pre-processed dataset

        Return:
        predictions (numpy array): the predicted values
        """
        groups = kwargs.get('groups')
        y_pred = np.zeros(X.shape[0])

        for bucket, stacking_regressor in self.stacking_regressors_.items():
            selector = (bucket == groups).flatten()
            if not np.any(selector):
                continue
            y_pred[selector] = stacking_regressor.predict(X[selector])

        return y_pred

    def set_params(self, **params):
        self.params = params
        return self

    def get_config(self, deep=True):
        return self.params

class BucketDoubleLevelStackingRegressor(BucketStackingRegressor):
    def _construct_stacking_regressor(self, X, y, trained_candidates):
        final_layer_base_models = [
            ('EXF', ExtraTreesRegressor()),
            ('KSVR', make_pipeline(Nystroem(), SVR())),
            ('RFR', RandomForestRegressor()),
        ]
        final_layer_lgbm = StackingRegressor(estimators=final_layer_base_models, final_estimator=lgb.LGBMRegressor(n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1, random_state=42), n_jobs=-1)
        
        base_models = [(name, v["model"]) for name, v in trained_candidates.items()]
        s_regressor = StackingRegressor(estimators=base_models, final_estimator=final_layer_lgbm, n_jobs=-1).fit(X, y)
        return s_regressor