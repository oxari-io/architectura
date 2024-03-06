# pip install autoimpute
import time
from lightgbm import LGBMClassifier

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from base.common import OxariFeatureReducer, OxariPreprocessor, OxariScopeTransformer
from imputers.categorical import HybridCategoricalStatisticsImputer
from imputers.core import BaselineImputer, DummyImputer
from pipeline import DefaultPipeline
from preprocessors import IIDPreprocessor
from preprocessors.core import BaselinePreprocessor
from preprocessors.helper.custom_cat_normalizers import (
    CountryCodeCatColumnNormalizer,
    LinkTransformerCatColumnNormalizer,
    OxariCategoricalNormalizer,
    IndustryNameCatColumnNormalizer,
    SectorNameCatColumnNormalizer,
)
from scope_estimators import SupportVectorEstimator, FastSupportVectorEstimator
from base.helper import ArcSinhTargetScaler, BucketScopeDiscretizer, DummyTargetScaler, LogTargetScaler
from base.run_utils import (
    get_default_datamanager_configuration,
    get_remote_datamanager_configuration,
    get_small_datamanager_configuration,
)
from feature_reducers import PCAFeatureReducer, DummyFeatureReducer
from imputers import (RevenueQuantileBucketImputer)
from datasources import S3Datasource
from sklearn.preprocessing import PowerTransformer, RobustScaler, minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it
from sklearn.ensemble import RandomForestClassifier
from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator, MiniModelArmyEstimator
from scope_estimators.mma.classifier import LGBMBucketClassifier, RandomForesBucketClassifier
import optuna

N_TRIALS = 20
N_STARTUP_TRIALS = 40


def prepare_data_for_classification(preprocessor: OxariPreprocessor, feature_reducer: OxariFeatureReducer, target_scaler: OxariScopeTransformer,
                                    discretizer: BucketScopeDiscretizer, X: pd.DataFrame, y: pd.Series):
    X_ = preprocessor.fit_transform(X, y)
    X_prep = feature_reducer.fit_transform(X_)
    y_prep = target_scaler.transform(y)
    discretizer = discretizer.fit(X_prep, y_prep)
    y_binned_train = discretizer.transform(y_prep)
    return X_prep, y_binned_train


# def optimize_rf(X_val, y_val, X_train, y_train):
#     model = RandomForestClassifier()
#     # Optuna
#     study: optuna.Study = optuna.create_study(direction="maximize",
#                                               sampler=optuna.samplers.TPESampler(n_startup_trials=40, warn_independent_sampling=False))
#     study.set_user_attr("model", model.__class__.__name__)

#     def retrieve_param_space(trial: optuna.Trial, X_train, y_train, X_val, y_val):
#         model.set_params(**{
#             'max_depth': trial.suggest_int('max_depth', 3, 21, step=3),
#             'n_estimators': trial.suggest_int("n_estimators", 100, 1500, step=100),
#         })
#         model.fit(X_train, y_train)

#         y_hat = model.predict(X_val)
#         precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_hat, average="weighted")
#         return f1

#     study.optimize(lambda trial: retrieve_param_space(trial, X_train, y_train, X_val, y_val), n_trials=20)

#     best_params: optuna.Trial = study.best_params
#     best_value = study.best_value
#     model.set_params(**best_params)

#     df: pd.DataFrame = study.trials_dataframe()
#     return model, best_value, best_params, df


def pspace_rf(trial: optuna.Trial):
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 40, step=3),
        'n_estimators': trial.suggest_int("n_estimators", 100, 1500, step=100),
    }


def pspace_lgbm(trial: optuna.Trial):
    return {
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 40, step=3),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int("n_estimators", 100, 500, step=100),
    }


def optimize(model, pspace_fn, X_val, y_val, X_train, y_train):
    # Optuna
    study: optuna.Study = optuna.create_study(direction="maximize",
                                              sampler=optuna.samplers.TPESampler(n_startup_trials=40, warn_independent_sampling=False))
    study.set_user_attr("model_name", model.__class__.__name__)

    def objective(trial):
        return precision_recall_fscore_support(y_val, model.set_params(**pspace_fn(trial)).fit(X_train, y_train.flatten()).predict(X_val), average="weighted")[2]

    study.optimize(objective, n_trials=20)

    best_params: optuna.Trial = study.best_params
    best_value = study.best_value

    df: pd.DataFrame = study.trials_dataframe()
    return model, best_value, best_params, df



if __name__ == "__main__":
    # TODO: Finish this experiment by adding LinearSVR
    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    dataset: OxariDataManager = get_small_datamanager_configuration(0.5).run()
    preprocessor: OxariPreprocessor = BaselinePreprocessor(fin_transformer=PowerTransformer()).set_imputer(BaselineImputer())
    feature_reducer = DummyFeatureReducer()
    target_scaler = ArcSinhTargetScaler()
    discretizer = BucketScopeDiscretizer(10)

    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    X, y = SPLIT_1.train
    X_val, y_val = SPLIT_1.val

    X_train, y_train = prepare_data_for_classification(preprocessor, feature_reducer, target_scaler, discretizer, X, y)
    X_val, y_val = prepare_data_for_classification(preprocessor, feature_reducer, target_scaler, discretizer, X_val, y_val)

    results_rf = optimize(RandomForestClassifier(), pspace_rf, X_val, y_val, X_train, y_train)
    results_lgbm = optimize(LGBMClassifier(), pspace_lgbm, X_val, y_val, X_train, y_train)

    fname = __loader__.name.split(".")[-1]
    pd.concat([results_rf[3], results_lgbm[3]], axis=0).to_csv(f"local/eval_results/{fname}.csv")
    print("=========================================================================================")
    print("=========================================================================================")

