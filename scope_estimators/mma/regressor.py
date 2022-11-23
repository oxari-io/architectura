import re
import json
import numpy as np
import pandas as pd
import pickle
import functools
import operator
import shap
import xgboost as xgb
import io
import warnings
import optuna
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Dict
from base.common import OxariEvaluator, OxariMixin, OxariOptimizer, OxariRegressor
# from sklearn.metrics import root_mean_squared_error as rmse
# from sklearn.metrics import mean_absolute_percentage_error as mape
# from model.misc.metrics import mape

# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# from ngboost import NGBRegressor
# from scipy import stats
# from pprint import pprint

from tqdm import tqdm
from pmdarima.metrics import smape

# from model.misc.mappings import NumMapping as Mapping
# from model.misc.metrics import adjusted_r_squared
# from model.misc.hyperparams_tuning import tune_hps_regressors

from pathlib import Path
# from model.abstract_base_class import MLModelInterface

# from model.misc.ML_toolkit import add_bucket_label, check_scope

OBJECT_DIR = Path("model/objects")
METRICS_DIR = Path("model/metrics")
DATA_DIR = Path("model/data")
OPTUNA_DIR = Path("model/optuna")


class RegressorOptimizer(OxariOptimizer):
    def __init__(self, n_trials=2, n_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_trials = n_trials
        self.num_startup_trials = n_startup_trials
        self.sampler = sampler or optuna.samplers.TPESampler(n_startup_trials=self.num_startup_trials, warn_independent_sampling=False)
        self.scope = None
        self.bucket_specific = None

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
        models = [
            # ("GBR", GradientBoostingRegressor), # Is fundamentally the same as XGBOOST but XGBoost is better - https://stats.stackexchange.com/a/282814/361976
            ("RFR", RandomForestRegressor),
            ("XGB", xgb.XGBRegressor),
        ]
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
            for name, Model in models:

                # print(f"Training {name} ... ")

                study = optuna.create_study(
                    study_name=f"regressor_{name}_hp_tuning",
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(n_startup_trials=self.num_startup_trials, warn_independent_sampling=False),
                )
                study.optimize(
                    lambda trial: self.score_trial(trial, name, X_train[selector_train], y_train[selector_train].values, X_val[selector_val], y_val[selector_val].values),
                    n_trials=self.num_trials,
                    show_progress_bar=False)

                candidates[bucket_name][name]["best_params"] = study.best_params
                candidates[bucket_name][name]["Model"] = Model
                info[bucket_name][name] = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        return {"candidates":candidates}, info

    def score_trial(self, trial:optuna.Trial, regr_name:str, X_train, y_train, X_val, y_val):
        # TODO: add docstring here

        if regr_name == "GBR":

            param_space = {
                # The number of boosting stages to perform
                "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=200),
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
                'n_estimators': trial.suggest_int("n_estimators", 100, 900, 200),
                # 'criterion': trial.suggest_categorical('criterion', ['squared_error', 'mae']),
                # Whether bootstrap samples are used when building trees
                'bootstrap': trial.suggest_categorical('bootstrap', ['True', 'False']),
                # The maximum depth of the tree.
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),
                # The number of features to consider when looking for the best split
                'max_features': trial.suggest_categorical('max_features', [None, 'sqrt']),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 12, 2),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5, 1),
                # Grow trees with max_leaf_nodes in best-first fashion.
                # 'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 1, 100),
                "n_jobs": -1
            }

            model = RandomForestRegressor(**param_space)
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
                'n_estimators': trial.suggest_int("n_estimators", 100, 900, 200),

                # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
                'max_depth': trial.suggest_int('max_depth', 3, 21, 3),

                # 'random_state': trial.suggest_categorical('random_state', [2020]),
                # If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5, 1),
            }

            model = xgb.XGBRegressor(**param_space)
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

        return smape(y_val, y_pred)

    # def build_voting_regressor(self, X_train, y_train, X_val, y_val):
    #     """
    #     For each model, it makes an optuna study in order to explore the best hyperparameters
    #     The best hyperparameters are passed to each model for training
    #     Performance scores of each individual model is used as weights for the voting regressor

    #     Parameters:
    #     X_train (numpy array): training data (features)
    #     y_train (numpy array): training data (targets)
    #     X_test (numpy array): testing data (features)
    #     y_test (numpy array): testing data (target)
    #     """

    #     # selecting the models that will be trained to build the voting regressor --> tuple(name, model)
    #     models = [("GBR", GradientBoostingRegressor()), ("RFR", RandomForestRegressor(n_jobs=-1)), ("XGB", xgb.XGBRegressor(n_jobs=-1, base_score=0.5))]

    #     # self.vanilla_voting_regressor = VotingRegressor(estimators=models, n_jobs=-1)

    #     # scores will be used to compute weights for voting mechanism
    #     scores = []

    #     # candidates will be given to VotingRegressor
    #     candidates = []

    #     # do hyperparams tuning for each of the three models --> OPTUNA!
    #     for name, model in models:

    #         # print(f"Training {name} ... ")

    #         best_hps = self.tune_hp(name, X_train, y_train, X_val, y_val, num_startup_trials=self.n_startup_trials, n_trials=self.n_trials)

    #         model.set_params(**best_hps)

    #         # loop through the models and train individually
    #         model.fit(X_train, y_train)

    #         # calculate the score of each individual model to weight voting mechanism
    #         y_pred = model.predict(X_val)

    #         model_score = smape(y_val, y_pred)

    #         # the lower the smape the better
    #         scores.append(100 - model_score)

    #         # append a tuple ("name", model) to then give it to VotingRegressor
    #         candidates.append((name, model))

    #     return VotingRegressor(estimators=candidates, weights=scores, n_jobs=-1)





class BucketRegressor(OxariMixin, OxariRegressor):
    # TODO: add docstring
    def __init__(self, n_buckets=10):
        # self.scope = check_scope(scope)
        self.n_buckets = n_buckets
        self.voting_regressors: Dict[int, VotingRegressor] = {}

    def fit(self, X, y, **kwargs):
        """
        The main training loop that calls all the other functions
        Subsets data, splits in training, test and validation, builds and trains voting regressor, computes error metrics

        """
        groups = kwargs.get('groups')
        regressor_kwargs = kwargs.get("candidates")
        trained_candidates = {}
        for bucket, candidates_data in regressor_kwargs.items():
            selector = groups == bucket
            X_train, X_val, y_train, y_val = train_test_split(X[selector], y[selector], test_size=0.3)
            for name, candidate_data in candidates_data.items():
                best_params = candidate_data.get("best_params")
                ModelConstructor = candidate_data.get("Model")
                model = ModelConstructor(**best_params).fit(X_train, y_train)

                # calculate the score of each individual model to weight voting mechanism
                y_pred = model.predict(X_val)

                model_score = smape(y_val, y_pred)

                # the lower the smape the better
                candidate_data["score"] = 100 - model_score
                candidate_data["model"] = model
                trained_candidates[name] = candidate_data

            weights = [v["score"] for _, v in trained_candidates.items()]
            models = [(name, v["model"]) for name, v in trained_candidates.items()]
            self.voting_regressors[bucket] = VotingRegressor(estimators=models, weights=weights, n_jobs=-1).fit(X[selector], y[selector])

        return self

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        best_params, info = self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)
        return best_params, info
    
    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred)
    
    def predict(self, X:pd.DataFrame, **kwargs):
        """
        Voting regressor computes prediction

        Parameters:
        data (pandas.DataFrame): pre-processed dataset

        Return:
        predictions (numpy array): the predicted values
        """
        groups = kwargs.get('groups')
        y_pred = np.zeros(X.shape[0])

        for bucket, voting_regressor in self.voting_regressors.items():
            selector = (bucket == groups).flatten()
            if not np.any(selector):
                continue
            y_pred[selector] = voting_regressor.predict(X[selector])

        return y_pred

    # def subset_data(self, data):
    #     """
    #     It subsets the data based on the bucket label

    #     Parameters:
    #     data (pandas.DataFrame): pre-processed dataset

    #     Return:
    #     data (pandas.DataFrame): data subset by bucket label

    #     """
    #     # subsetting dataframe
    #     # returning only the portion of the data where group_label == bucket_specific
    #     return data.loc[data[f"group_label_{self.scope}"] == self.bucket_specific]

    # def train_test_val_split(self, data, split_size_test=None, split_size_val=None):
    #     """
    #     Splitting the data in trianing, testing, and validation sets
    #     with a splitting threshold of split_size_test, and split_size_val respectively

    #     Parameters:
    #     data (pandas.DataFrame): pre-processed dataset
    #     split_size_test (float): splitting threshold between training set and testing + validation sets
    #     split_size_val (float): splitting threshold between testing set and validation set

    #      Return:
    #     X_train (numpy array): training data
    #     y_train (numpy array): training data
    #     X_train_full (numpy array): not splitted training data
    #     y_train_full (numpy array): not splitted training data
    #     X_test (numpy array): testing data
    #     y_test (numpy array): testing data
    #     X_val (numpy array): validation data
    #     y_val (numpy array): validation data

    #     """

    #     # excluding -1 as they are not valid targets
    #     data = data.loc[data[self.scope] != -1]

    #     X, y = data.drop(columns=self.list_of_skipped_columns), data[self.scope]

    #     # verbose
    #     print(f"Number of datapoints: shape of y {y.shape}, shape of X {X.shape}")

    #     X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=split_size_test, random_state=42)

    #     # splitting further - train and validation sets will be used for hyperparameter-optimization; test set will be used for performance assesment
    #     X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=split_size_val, random_state=42)

    #     return X_train, y_train, X_train_full, y_train_full, X_test, y_test, X_val, y_val

    # def run_shap_analysis(self, X_test):

    #     # shap.initjs()
    #     print("DOING SHAPY")

    #     # feature_names = ["sector_name" , "ppe", "industry_name", "rd_expenses", "employees"]
    #     # # print(feature_names)
    #     feature_names = self.columns
    #     print(feature_names)

    #     model = self.voting_regressors.named_estimators_["XGB"]

    #     explainer = shap.TreeExplainer(model)

    #     shap_values = explainer(X_test)

    #     f = plt.figure()
    #     # shap.plots.beeswarm(shap_values)
    #     shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="violin")
    #     # print("finished plotting shapyy")
    #     f.savefig("./model/misc/XGB_summary_plot_scope1.png")

    # @staticmethod
    # def get_params(models):
    #     params = {}
    #     for m in models:
    #         name_ = str(m.__class__).split(".")[-1].rstrip("'>")
    #         params[name_] = m.get_params()

    #     return params

    # @staticmethod
    # def tune_hp(model, X_train, y_train, params, hyper_p, scope):
    #     search_space = {}
    #     for hp in hyper_p:
    #         if hp in params:
    #             print(f"tuning {hp}")
    #             search_space[hp] = hyper_p[hp]

    #     # by default cv --> 5
    #     random_search = RandomizedSearchCV(estimator = model, param_distributions=search_space, verbose = 1, n_jobs = -1).fit(X_train, y_train)
    #     # grid_search = GridSearchCV(estimator = model, param_grid=search_space, verbose = 1).fit(X_train, y_train)

    #     # return best hyper param
    #     # print("BEST hyper parameters", random_search.best_params_)
    #     # name_ = str(model.__class__).split(".")[-1].rstrip("'>")
    #     # with open(f"model/data/hps_{scope}.txt", "a+") as file:
    #     #     line = name_ + str(random_search.best_params_) + "\n\n"
    #     #     file.write(line)
    #     return random_search.best_params_

    # @staticmethod
    # def train_regressors(model, X_train, X_val, y_train, y_val, params, hyper_p, scope, with_hp = False, score_func = mean_squared_error):
    #     """
    #     inputs:
    #     model, data

    #     output:
    #     model_trained, model_score
    #     """

    #     # best_hparams = ModelML.tune_hp(model, X_train, y_train, params, hyper_p, scope)
    #     name_ = str(model.__class__).split(".")[-1].rstrip("'>")
    #     if with_hp:
    #         # read best hp from text file
    #         with open (f"model/data/hps_{scope}.txt") as handle:
    #             print("reading hps txt")
    #             hp_list = handle.readlines()

    #         for elem in hp_list:
    #             if name_ in elem:
    #                 #print("found it", elem)
    #                 best_hparams = json.loads(re.search("(?!\w+).+", elem).group(0))
    #                 #print(best_hparams)

    #         # set best hyperparams
    #         model = model.set_params(**best_hparams)

    #     # print("training with best hyperparams")
    #     model.fit(X_train, y_train)

    #     y_predicted = model.predict(X_val)

    #     model_score = score_func(y_predicted, y_val, squared = False)

    #     #print(f"model: {} {model_score})

    #     return model, 1/(model_score+0.1)

    # def process_regressors(self, data, error_table):

    #     scores = []
    #     candidates = []
    #     models = [GradientBoostingRegressor(), RandomForestRegressor(), KNeighborsRegressor(), xgb.XGBRegressor()]

    #     for lbl in [x for x in range(self.n_buckets)]:

    #         # subset data for each label
    #         X, y[y[f"label_group_{self.scope}" == lbl]] = add_bucket_label_and_split(data)

    #         X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3)

    #         # validation set is needed to calculate score(weights) for VotingClassifier
    #         X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.215)

    #         for m in tqdm(models):
    #             name_ = str(m.__class__).split(".")[-1].rstrip("'>")
    #             # the following will also perform hyperparameter tuning
    #             model, score_ = ModelML.train_regressors(m, X_train, X_val, y_train, y_val, params[name_], self.hyper_params, self.scope, with_hp=self.use_hp)
    #             print(f"\n training {name_}: {score_} ")
    #             candidates.append((name_, model))
    #             scores.append(score_)

    #             scores = np.array(scores) / sum(scores)
    #             #print(scores)
    #             self.ensemble = VotingRegressor(estimators=candidates, weights=scores, n_jobs=-1)
    #             self.ensemble.fit(X_train_full, y_train_full)

    #     params = ModelML.get_params(models)

    # X_train_full, X_test, y_train_full, y_test = self.train_test_split_by_scope(test_size=0.3, data=data)

    # # split full train data into validation and train
    # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.215)

    # for m in tqdm(models):
    #     name_ = str(m.__class__).split(".")[-1].rstrip("'>")
    #     # the following will also perform hyperparameter tuning
    #     model, score_ = ModelML.train_model(m, X_train, X_val, y_train, y_val, params[name_], self.hyper_params, self.scope, with_hp=self.use_hp)
    #     print(f"\n training {name_}: {score_} ")
    #     candidates.append((name_, model))
    #     scores.append(score_)

    # fitting the full training data to the VotingRegressor
    # self.columns = X_train_full.columns.to_list()

    # print("DOING ERROR METRICS")
    # error_table = pd.read_csv(DATA_DIR / "error_metrics_oldModel.csv")
    # y_pred_ensemble = self.ensemble.predict(X_test)
    # print("SMAPE: " , smape(y_test, y_pred_ensemble))

    # def conduct_error_metrics():
    # df = pd.DataFrame(columns=["model", "scope", "sMAPE", "adjR2", "MAE", "RMSE"])
    # df.loc[0, "model"] = "OXARI-ENSEMBLE"

    # error_table = pd.DataFrame(columns=["model", "scope", "sMAPE", "adjR2", "MAE", "RMSE"])

    # print("Error metrics ENSEMBLE")
    # error_ensemble = {"model" : "OXARI", "scope" : self.scope, "sMAPE" : smape(y_test, y_pred_ensemble),
    #                   "adjR2" : adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_ensemble)) ,
    #                   "MAE" : mean_absolute_error(y_test, y_pred_ensemble), "RMSE" : mean_squared_error(y_test, y_pred_ensemble, squared=False)}

    # error_table = pd.read_csv("error_metrics_oldModel.csv")

    # error_table = error_table.append(error_ensemble, ignore_index = True)

    # xgboost = xgb.XGBRegressor()
    # xgboost.fit(X_train_full, y_train_full)
    # y_pred_xgb = xgboost.predict(X_test)

    # print("Error metrics XGB")

    # error_xgb = {"model" : "vannilla_XGB", "scope" : self.scope, "sMAPE" : smape(y_test, y_pred_xgb),
    #                   "adjR2" : adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_xgb)) ,
    #                   "MAE" : mean_absolute_error(y_test, y_pred_xgb), "RMSE" : mean_squared_error(y_test, y_pred_xgb, squared=False)}

    # error_table = error_table.append(error_xgb, ignore_index = True)

    # knn = KNeighborsRegressor()
    # knn.fit(X_train_full, y_train_full)
    # y_pred_knn = knn.predict(X_test)

    # print("Error metrics KNN")

    # error_knn = {"model" : "vannilla_KNN", "scope" : self.scope, "sMAPE" : smape(y_test, y_pred_knn),
    #                   "adjR2" : adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_knn)) ,
    #                   "MAE" : mean_absolute_error(y_test, y_pred_knn), "RMSE" : mean_squared_error(y_test, y_pred_knn, squared=False)}

    # error_table = error_table.append(error_knn, ignore_index = True)

    # lr = LinearRegression().fit(X_train_full, y_train_full)
    # y_pred_lr = lr.predict(X_test)

    # error_lr = {"model" : "LR", "scope" : self.scope, "sMAPE" : smape(y_test, y_pred_lr),
    #                   "adjR2" : adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_lr)) ,
    #                   "MAE" : mean_absolute_error(y_test, y_pred_lr), "RMSE" : mean_squared_error(y_test, y_pred_lr, squared=False)}

    # error_table = error_table.append(error_lr, ignore_index = True)

    # error_table.to_csv(DATA_DIR/"error_metrics_oldModel.csv", index=False)

    # print("DONE ERROR METRICS")

    # with open ("error_metrics_oldModel.txt", "a+") as file:
    #     file.write(f"OXARI ENSEMBLE - {self.scope} \n")
    #     file.write(f"sMAPE : {smape(y_test, y_pred_ensemble)}")
    #     file.write("\n")
    #     file.write(f"adj R2 : {adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_ensemble))}")
    #     file.write("\n")
    #     file.write(f"MAE : {mean_absolute_error(y_test, y_pred_ensemble)}")
    #     file.write("\n")
    #     file.write(f"RMSE : {mean_squared_error(y_test, y_pred_ensemble, squared=False)}")
    #     file.write("\n")
    #     file.write("\n")
    #     file.write("\n")

    # with open ("error_metrics_oldModel.txt", "a+") as file:
    #     file.write(f"XGBoost - {self.scope} \n")
    #     file.write(f"sMAPE : {smape(y_test, y_pred_xgb)}")
    #     file.write("\n")
    #     file.write(f"adj R2 : {adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_xgb))}")
    #     file.write(f"MAE : {mean_absolute_error(y_test, y_pred_xgb)}")
    #     file.write("\n")
    #     file.write(f"RMSE : {mean_squared_error(y_test, y_pred_xgb, squared=False)}")
    #     file.write("\n")
    #     file.write("\n")

    # with open ("error_metrics_oldModel.txt", "a+") as file:
    #     file.write(f"vanilla KNN - {self.scope} \n")
    #     file.write(f"sMAPE : {smape(y_test, y_pred_knn)}")
    #     file.write("\n")
    #     file.write(f"adj R2 : {adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_knn))}")
    #     file.write("\n")
    #     file.write(f"MAE : {mean_absolute_error(y_test, y_pred_knn)}")
    #     file.write("\n")
    #     file.write(f"RMSE : {mean_squared_error(y_test, y_pred_knn, squared=False)}")
    #     file.write("\n")
    #     file.write("\n")

    # lr = LinearRegression().fit(X_train_full, y_train_full)
    # y_pred_lr = lr.predict(X_test)

    # with open ("error_metrics_oldModel.txt", "a+") as file:
    #     file.write(f"LR - {self.scope} \n")
    #     file.write(f"sMAPE : {smape(y_test, y_pred_lr)}")
    #     file.write("\n")
    #     file.write(f"adj R2 : {adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_lr))}")
    #     file.write("\n")
    #     file.write(f"MAE : {mean_absolute_error(y_test, y_pred_lr)}")
    #     file.write("\n")
    #     file.write(f"RMSE : {mean_squared_error(y_test, y_pred_lr, squared=False)}")
    #     file.write("\n")
    #     file.write("\n")

    # lr_bin = LinearRegression().fit(X_train_full["revenue"].values.reshape(-1, 1), y_train_full)

    # y_pred_lr_bin = lr_bin.predict(X_test["revenue"].values.reshape(-1, 1))

    # with open ("error_metrics_oldModel.txt", "a+") as file:
    #     file.write(f"LR-revenue: {self.scope}")
    #     file.write("\n")
    #     file.write(f"sMAPE : {smape(y_test, y_pred_lr_bin)}")
    #     file.write("\n")
    #     file.write(f"adj R2 : {adjusted_r_squared(X_train_full, y_train_full, r2_score(y_test, y_pred_lr_bin))}")
    #     file.write("\n")
    #     file.write("\n")
    #     file.write("\n")

    # if self.scope == "scope_1":
    #     print("running SHAP")
    #     self.run_shap_analysis(X_test)

    # def process_predict(self, prediction_data):
    #     X, y = prediction_data.drop(columns = self.list_of_skipped_columns), prediction_data[self.scope]

    #     estimated_scope = prediction_data[["isin", "year", self.scope]]

    #     estimated_scope.loc[: , self.scope] = np.exp(self.ensemble.predict(X[self.columns]))-1

    #     return estimated_scope

    # @staticmethod
    # def train_autoML(X_train, y_train, X_test, y_test):

    # metric: A metric which we want to optimize

    # scoring_function: One or more metrics which we want to evaluate the model on

    # Available REGRESSION autosklearn.metrics.*:
    # *mean_absolute_error
    # *mean_squared_error
    # *root_mean_squared_error
    # *mean_squared_log_error
    # *median_absolute_error
    # *r2

    # By default, the regressor will optimize the R^2 metric.
    # autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)

    # automl = autosklearn.regression.AutoSklearnRegressor(
    # time_left_for_this_task=100,
    # ensemble_size = 20,
    # ensemble_nbest = 10,
    # metric = mse,
    # include = {
    # #"regressor" : ["gradient_boosting"],
    # 'data_preprocessor' : ['NoPreprocessing'],
    # 'feature_preprocessor': ["no_preprocessing"]},
    # initial_configurations_via_metalearning=0
    # )

    # automl.fit(X = X_train, y = y_train, X_test = X_test, y_test = y_test)
    # pprint(automl.leaderboard(detailed = True))

    # print()

    #ensemble_dict = automl.show_models()
    # print(ensemble_dict)

    # def calculate_error_metrics(y_test, X_train, y_train, X_test):
    #     y_pred_ensemble = self.ensemble.predict(X_test)

    #     with open ("error_metrics.txt", "a+") as file:
    #         file.write("~ ENSEMBLE scope: ", self.scope, "~")
    #         file.write("sMAPE : ", smape(y_test, y_pred_ensemble))
    #         file.write("R2 : " , r2_score(y_test, y_pred_ensemble))
    #         file.write("\n")
    #         file.write("\n")

    #     model = self.ensemble.named_estimators_["XGBRegressor"]
    #     y_pred_xgb = model.predict(X_test)

    #     with open ("error_metrics.txt", "a+") as file:
    #         file.write("~ XGBR scope: ", self.scope, "~")
    #         file.write("sMAPE : ", smape(y_test, y_pred_xgb))
    #         file.write("R2 : " , r2_score(y_test, y_pred_xgb))
    #         file.write("\n")
    #         file.write("\n")

    #     lr = LinearRegression().fit(X_train, y_train)
    #     y_pred_lr = lr.predict(X_test)

    #     with open ("error_metrics.txt", "a+") as file:
    #         file.write("~ LR scope: ", self.scope, "~")
    #         file.write("sMAPE : ", smape(y_test, y_pred_lr))
    #         file.write("R2 : " , r2_score(y_test, y_pred_lr))
    #         file.write("\n")
    #         file.write("\n")

    #     lr_bin = LinearRegression().fit(X_train["revenue"].reshape(-1, 1), y_train)

    #     y_pred_lr_bin = lr_bin.predict(X_test)

    #     with open ("error_metrics.txt", "a+") as file:
    #         file.write("~ LR_bins scope: ", self.scope, "~")
    #         file.write("sMAPE : ", smape(y_test, y_pred_lr_bin))
    #         file.write("adj R2 : " , r2_score(y_test, y_pred_lr_bin))
    #         file.write("\n")
    #         file.write("\n")

    #     print("DONE ERROR METRICS")


# class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):

#     def __init__(self, **kwargs):
#         """ This preprocessors does not change the data """
#         # Some internal checks makes sure parameters are set
#         for key, val in kwargs.items():
#             setattr(self, key, val)

#     def fit(self, X, Y=None):
#         return self

#     def transform(self, X):
#         return X

#     @staticmethod
#     def get_properties(dataset_properties=None):
#         return {
#             'shortname': 'NoPreprocessing',
#             'name': 'NoPreprocessing',
#             'handles_regression': True,
#             'handles_classification': True,
#             'handles_multiclass': True,
#             'handles_multilabel': True,
#             'handles_multioutput': True,
#             'is_deterministic': True,
#             'input': (SPARSE, DENSE, UNSIGNED_DATA),
#             'output': (INPUT,)
#         }

#     @staticmethod
#     def get_hyperparameter_search_space(dataset_properties=None):
#         return ConfigurationSpace()  # Return an empty configuration as there is None
