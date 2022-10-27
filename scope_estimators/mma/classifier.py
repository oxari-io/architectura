import pandas as pd
import pickle
import numpy as np
import joblib
import optuna

from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
# from model.abstract_base_class import MLModelInterface

# from model.misc.hyperparams_tuning import tune_hps_classifier
# from model.misc.ML_toolkit import add_bucket_label,check_scope

from pathlib import Path

from base.common import OxariClassifier, OxariEvaluator, OxariMixin, OxariOptimizer

OBJECT_DIR = Path("model/objects")
DATA_DIR = Path("model/data")
METRICS_DIR = Path("model/metrics")
OPTUNA_DIR = Path("model/optuna")


class ClassifierOptimizer(OxariOptimizer):
    def __init__(self, num_trials = 1, num_startup_trials = 1, sampler=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_trials = num_trials
        self.num_startup_trials = num_startup_trials
        self.sampler = sampler or optuna.samplers.CmaEsSampler(n_startup_trials=self.num_startup_trials, warn_independent_sampling=False)

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
            study_name=f"classifier_hp_tuning_{self.scope}",
            direction="maximize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.tune_hps_classifier(trial, X_train, y_train, X_val, y_val), n_trials=self.n_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        df.to_csv(OPTUNA_DIR / f"df_optuna_hps_CL_{self.scope}_buckets_{self.n_buckets}.csv", index=False)

        # save the study so that we can plot the results
        # joblib.dump(study, OPTUNA_DIR / f"optuna_hps_CL_{self.scope}_{self.n_buckets}_buckets.pkl")

        return study.best_params

    def tune_hps_classifier(self, trial, X_train, y_train, X_val, y_val):

        # TODO: add docstring here pls 

        # cl_name = trial.suggest_categorical("classifier", ["RF", "XGB"])
        cl_name = "RF"

        if cl_name == "RF":
            # min_impurity_decrease,  max_leaf_nodes, min_weight_fraction_leaf, warm_start
            param_space = {'n_estimators': trial.suggest_int("n_estimators", 100, 1000, 100), #100, 200, 300
                        # 'max_depth': trial.suggest_int("max_depth", 5, 70, 5),
                        'min_samples_split': trial.suggest_int("min_samples_split", 2, 12, 2),
                        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5, 1),
                        # 'max_leaf_nodes' : trial.suggest_int("max_leaf_nodes", 1, 40, 2),
                        # 'bootstrap': trial.suggest_categorical("bootstrap", [True, False]),
                        # 'max_features': trial.suggest_categorical("max_features", [None, "sqrt"]),
                            # 'criterion': trial.suggest_categorical('criterion', ['mse', 'mae']),
                            # Whether bootstrap samples are used when building trees
                            'bootstrap': trial.suggest_categorical('bootstrap',['True','False']),
                            # The maximum depth of the tree.
                            'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                            # The number of features to consider when looking for the best split
                            # 'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt','log2']),
                            # Grow trees with max_leaf_nodes in best-first fashion.
                            # 'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 1, 20, 1),
                        'n_jobs': -1}

            cl = RandomForestClassifier(**param_space)

            # rfecv = RFE(
            #     estimator=RandomForestClassifier(),
            #     # n_jobs = -1,
            #     verbose = 2,
            #     # min_features_to_select= 10,
            #     step=5,

            # )
            # _ = rfecv.fit(X_train, y_train)
            # print("Training RF_CL")
            # cl.fit(X_train.loc[:,rfecv.support_], y_train)
            cl.fit(X_train, y_train)

        else:
            # TODO: could be adding more hps
            param_space = {
                            "max_depth": trial.suggest_int("max_depth", 3, 30, 3),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.31, step=0.05),
                        "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.1),
                        # "colsample_bytree" : trial.suggest_float("colsample_bytree",0.6, 1, step = 0.2),
                        # "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.6, 1, step  = 0.2),
                        "n_estimators":  trial.suggest_int("n_estimators", 200, 1000, 200),
                        "n_jobs": -1}

            cl = xgb.XGBClassifier(**param_space)

            # rfecv = RFE(
            #     estimator=xgb.XGBClassifier(),
            #     # n_jobs = -1,
            #     verbose = 1,
            #     # min_features_to_select= 15,
            #     step=5,
            # )
            # _ = rfecv.fit(X_train, y_train)
            # print("Training XGB_CL")

            # cl.fit(X_train.loc[:,rfecv.support_], y_train)
            cl.fit(X_train, y_train)

        y_pred = cl.predict(X_val)

        # choose weighted average
        f1 = f1_score(y_true=y_val, y_pred=y_pred, average="macro")
        # supp = precision_recall_fscore_support(y_true=y_test, y_pred = y_pred, average="micro")
        return f1


class ClassifierEvaluator(OxariEvaluator):
    def __init__(self, scope, n_buckets, **kwargs) -> None:
        super().__init__()
        self.scope = scope
        self.n_buckets = n_buckets

    def evaluate(self, y_test, y_pred, **kwargs):
        """

        Computes 3 flavors of accuracy: Vanilla, AdjacentLenient, AdjacentStrict

        Each accuracy computation is scope and buckets specific

        Appends and saves the results to model/metrics/error_metrics_class.csv

        """

        # print(
        #     f"VANILLA Accuracy for tuned model {balanced_accuracy_score(y_test, y_pred)}", "\n")

        # print(
        #     f"LENIENT Adj Accuracy for tuned model {self.lenient_adjacent_accuracy_score(y_test, y_pred)}", "\n")

        # print(
        #     f"STRICT Adj Accuracy for tuned model {self.strict_adjacent_accuracy_score(y_test, y_pred)}", "\n")

        return {
            "scope": self.scope,
            "n_buckets": self.n_buckets,
            "vanilla_acc": balanced_accuracy_score(y_test, y_pred),
            "adj_lenient_acc": self.lenient_adjacent_accuracy_score(y_test, y_pred),
            "adj_strict_acc": self.strict_adjacent_accuracy_score(y_test, y_pred)
        }

    def lenient_adjacent_accuracy_score(self, y_true, y_pred):
        # if true == 0 and pred == 1 --> CORRECT!
        # if true == 9 and pred == 8 --> CORRECT!
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sum(np.abs(y_pred - y_true) <= 1) / len(y_pred)

    def being_strict(self, y_true, y_pred):
        """
        # strict with top and bottom bucket
        # we want extreme buckets to always be correctly predicted
        # if true == 0 and pred == 1 --> WRONG!
        # if true == 9 and pred == 8 --> WRONG!

         5 buckets 0 to 4
         y_true = [0, 2, 4, 3, 1]
         y_pred = [1, 1, 3, 4, 0]
         FALSE, TRUE, FALSE, FALSE, FALSE

        """
        if y_true in [self.n_buckets - 1, 0]:
            return np.abs(y_true - y_pred) == 0
        elif y_pred in [self.n_buckets - 1, 0] and y_true in [self.n_buckets - 2, 1]:
            return np.abs(y_true - y_pred) == 0
        else:
            return np.abs(y_true - y_pred) <= 1

    def strict_adjacent_accuracy_score(self, y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        vfunc = np.vectorize(self.being_strict)

        return np.sum(vfunc(y_true, y_pred)) / len(y_pred)


class BucketClassifier(OxariClassifier, OxariMixin):
    def __init__(self, object_filename, scope, optimizer = None, evaluator = None ,use_hp=False, n_buckets=10):

        self.object_filename = object_filename

        # self.scope = check_scope(scope)
        self.scope = scope
        self.n_buckets = n_buckets

        self.columns = None

        self.optimizer = optimizer or ClassifierOptimizer()

        self.evaluator = evaluator or ClassifierEvaluator(scope = self.scope, n_buckets = self.n_buckets)

        # self.verbose = verbose


        self.cl = RandomForestClassifier()

        self.list_of_skipped_columns = ['scope_1', 'scope_2', "scope_3", 'isin', "year", f"group_label_{self.scope}"]

    def fit(self, X, y, **kwargs) -> "OxariClassifier":
        """
        Splits the data into different buckets. Finds the best hyperparameters for the classifier 
        and trains the classifier given the best hyperparameters on the full training dataset (no validation).
        Computes the error metrics at the end on a test set.

        Parameters:
        data (pandas.DataFrame): pre-processed dataset
        """
        

        # data = add_bucket_label(data, self.scope, self.n_buckets)

        # X_train, y_train, X_train_full, y_train_full, X_test, y_test, X_val, y_val = self.train_test_val_split(data, split_size_test=0.2, split_size_val=0.4)

        # best_hps = self.optimize(X_train, y_train, X_val, y_val, num_startup_trials=self.n_startup_trials, n_trials=self.n_trials)
        self.cl = RandomForestClassifier(**kwargs) 

        self.cl.fit(X, y)
        
        return self

        

    def predict(self, X):
        """
        # TODO: rewrite this

        Classfies the companies in one of the 3 scopes using only a subset of columns, given a deepcopy of the dataframe
        Adds new columns in the dataframe with the scope label and its corresponding row with the predicted value

        Parameters:
        data (pandas.DataFrame): pre-processed dataset in pandas format from data pipeline, transformed after each prediction
        deepcopy_data (pandas.DataFrame): the dataframe that will be transformed during the prediction

        Return:
        data (pandas.DataFrame): original dataframe with attached columns for the 3 computed scopes
        """

        # X[f"group_label_{self.scope}"] = 

        # data[f"group_label_{self.scope}"] = self.cl.predict(deepcopy_data[deepcopy_data.columns.difference(self.list_of_skipped_columns[:-1])])
        return self.cl.predict(X)

