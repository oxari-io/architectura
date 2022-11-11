from typing import Union
import pandas as pd
import pickle
import numpy as np
import joblib
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
# from model.abstract_base_class import MLModelInterface
from sklearn.preprocessing import KBinsDiscretizer
# from model.misc.hyperparams_tuning import tune_hps_classifier
# from model.misc.ML_toolkit import add_bucket_label,check_scope

from pathlib import Path

from base.common import OxariClassifier, OxariEvaluator, OxariMixin, OxariOptimizer, OxariTransformer

OBJECT_DIR = Path("model/objects")
DATA_DIR = Path("model/data")
METRICS_DIR = Path("model/metrics")
OPTUNA_DIR = Path("model/optuna")


class ClassfierScopeDiscretizer(OxariTransformer):
    def __init__(self, n_buckets, prefix="bucket_", **kwargs) -> None:
        super().__init__()
        self.n_buckets = n_buckets
        self.prefix = prefix
        encode = kwargs.pop("encode", "ordinal")
        self.discretizer = KBinsDiscretizer(n_buckets, encode=encode, **kwargs)

    def fit(self, X, y=None):
        self.discretizer.fit(X.values[:,None])
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return self.discretizer.transform(X.values[:, None], **kwargs)

class ClassifierOptimizer(OxariOptimizer):
    def __init__(self, num_trials=2, num_startup_trials=1, sampler=None, **kwargs) -> None:
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
            study_name=f"classifier_hp_tuning",
            direction="maximize",
            sampler=self.sampler,
        )

        # running optimization
        # trials is the full number of iterations
        study.optimize(lambda trial: self.score_trial(trial, X_train, y_train, X_val, y_val), n_trials=self.num_trials, show_progress_bar=False)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        # df.to_csv(OPTUNA_DIR / f"df_optuna_hps_CL_{self.scope}_buckets_{self.n_buckets}.csv", index=False)

        # save the study so that we can plot the results
        # joblib.dump(study, OPTUNA_DIR / f"optuna_hps_CL_{self.scope}_{self.n_buckets}_buckets.pkl")

        return study.best_params, df

    def score_trial(self, trial:optuna.Trial, X_train, y_train, X_val, y_val):

        # TODO: add docstring here pls

        # cl_name = trial.suggest_categorical("classifier", ["RF", "XGB"])
        cl_name = "RF"

        if cl_name == "RF":
            # min_impurity_decrease,  max_leaf_nodes, min_weight_fraction_leaf, warm_start
            param_space = {
                'n_estimators': trial.suggest_int("n_estimators", 100, 1000, 100),  #100, 200, 300
                # 'max_depth': trial.suggest_int("max_depth", 5, 70, 5),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 12, 2),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5, 1),
                # 'max_leaf_nodes' : trial.suggest_int("max_leaf_nodes", 1, 40, 2),
                # 'bootstrap': trial.suggest_categorical("bootstrap", [True, False]),
                # 'max_features': trial.suggest_categorical("max_features", [None, "sqrt"]),
                # 'criterion': trial.suggest_categorical('criterion', ['mse', 'mae']),
                # Whether bootstrap samples are used when building trees
                'bootstrap': trial.suggest_categorical('bootstrap', ['True', 'False']),
                # The maximum depth of the tree.
                'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                # The number of features to consider when looking for the best split
                # 'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt','log2']),
                # Grow trees with max_leaf_nodes in best-first fashion.
                # 'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 1, 20, 1),
                'n_jobs': -1
            }

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
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000, 200),
                "n_jobs": -1
            }

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





class BucketClassifier(OxariClassifier, OxariMixin):
    def __init__(self, n_buckets=10, **kwargs):
        self.n_buckets = n_buckets
        self._estimator = RandomForestClassifier(**kwargs)

    def optimize(self,X_train, y_train, X_val, y_val, **kwargs):
        best_params, info = self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)
        return best_params, info
    
    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred)

    def fit(self, X, y, **kwargs) -> "OxariClassifier":
        self._estimator.set_params(**kwargs).fit(X, y.ravel())
        return self

    def predict(self, X, **kwargs):
        return self._estimator.predict(X)
