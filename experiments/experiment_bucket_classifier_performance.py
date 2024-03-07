# pip install autoimpute
import time
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
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
from base.helper import (
    ArcSinhTargetScaler,
    BucketScopeDiscretizer,
    DummyTargetScaler,
    LogTargetScaler,
)
from base.run_utils import (
    get_default_datamanager_configuration,
    get_remote_datamanager_configuration,
    get_small_datamanager_configuration,
)
from feature_reducers import PCAFeatureReducer, DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer
from datasources import S3Datasource
from sklearn.preprocessing import PowerTransformer, RobustScaler, minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it
from sklearn.ensemble import RandomForestClassifier
from scope_estimators.mini_model_army import (
    EvenWeightMiniModelArmyEstimator,
    MiniModelArmyEstimator,
)
from scope_estimators.mma.classifier import (
    LGBMBucketClassifier,
    RandomForesBucketClassifier,
)
import optuna

N_TRIALS = 40
N_STARTUP_TRIALS = 40
DATA_AMOUNT = 0.5


def prepare_data_for_classification(
    preprocessor: OxariPreprocessor,
    feature_reducer: OxariFeatureReducer,
    target_scaler: OxariScopeTransformer,
    discretizer: BucketScopeDiscretizer,
    X: pd.DataFrame,
    y: pd.Series,
):
    X_prep = feature_reducer.transform(preprocessor.transform(X))
    y_prep = target_scaler.transform(y)
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
        "max_depth": trial.suggest_int("max_depth", 3, 99, step=3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=100),
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
        "max_depth": trial.suggest_int("max_depth", 3, 99, step=3),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1500, step=100),
    }


def pspace_xgb(trial: optuna.Trial):
    param = {
        "booster": trial.suggest_categorical("booster", ["dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    return param


def optimize(model, pspace_fn, X_val, y_val, X_train, y_train):
    # Optuna
    study: optuna.Study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS, warn_independent_sampling=False),
    )
    study.set_user_attr("model_name", model.__class__.__name__)

    def objective(trial):
        return precision_recall_fscore_support(
            y_val,
            model.set_params(**pspace_fn(trial)).fit(X_train, y_train.flatten()).predict(X_val),
            average="weighted",
        )[2]

    study.optimize(objective, n_trials=N_TRIALS)

    best_params: optuna.Trial = study.best_params
    best_value = study.best_value

    df: pd.DataFrame = study.trials_dataframe()
    metrics = precision_recall_fscore_support(
        y_val,
        model.set_params(**best_params).fit(X_train, y_train.flatten()).predict(X_val),
    )

    return (
        model,
        best_value,
        best_params,
        df,
        pd.DataFrame(metrics, index=["precision", "recall", "f1", "support"]),
    )


if __name__ == "__main__":
    # TODO: Finish this experiment by adding LinearSVR
    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    dataset: OxariDataManager = get_small_datamanager_configuration(DATA_AMOUNT).run()

    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    X, y = SPLIT_1.train
    X_, y_ = SPLIT_1.val

    preprocessors: list[OxariPreprocessor] = [
        BaselinePreprocessor(fin_transformer=PowerTransformer()),
        IIDPreprocessor(fin_transformer=PowerTransformer()),
    ]

    imputers: list[OxariImputer] = [
        DummyImputer(),
        RevenueQuantileBucketImputer(),
        HybridCategoricalStatisticsImputer(),
    ]

    target_scalers: list[OxariScopeTransformer] = [
        LogTargetScaler(),
        ArcSinhTargetScaler(),
        DummyTargetScaler(),
    ]

    i = 0
    fname = __loader__.name.split(".")[-1]
    df_results = pd.DataFrame()
    df_metrics = pd.DataFrame()
    for preprocessor in preprocessors:
        for imputer in imputers:
            for target_scaler in target_scalers:
                preprocessor.set_imputer(imputer)
                feature_reducer = DummyFeatureReducer()
                discretizer = BucketScopeDiscretizer(10)

                # Makes sure everything is trained properly once.
                X_prep = feature_reducer.fit(preprocessor.fit_transform(X, y), y)
                discretizer.fit(X_prep, target_scaler.fit(X_prep, y).transform(y))

                # NOTE should not . Maybe that's the issue
                X_train, y_train = prepare_data_for_classification(preprocessor, feature_reducer, target_scaler, discretizer, X, y)
                X_val, y_val = prepare_data_for_classification(preprocessor, feature_reducer, target_scaler, discretizer, X_, y_)

                results_rf = optimize(RandomForestClassifier(), pspace_rf, X_val, y_val, X_train, y_train)
                results_lgbm = optimize(LGBMClassifier(), pspace_lgbm, X_val, y_val, X_train, y_train)
                results_xgb = optimize(XGBClassifier(), pspace_xgb, X_val, y_val, X_train, y_train)

                results = pd.concat(
                    [
                        results_rf[3].assign(model="RandomForest", best_value=results_rf[1]),
                        results_lgbm[3].assign(model="LGBM", best_value=results_lgbm[1]),
                        results_xgb[3].assign(model="XGB", best_value=results_xgb[1]),
                    ],
                    axis=0,
                )
                metrics = pd.concat(
                    [
                        results_rf[4].assign(model="RandomForest", best_value=results_rf[1]),
                        results_lgbm[4].assign(model="LGBM", best_value=results_lgbm[1]),
                        results_xgb[4].assign(model="XGB", best_value=results_xgb[1]),
                    ],
                    axis=0,
                )

                results = results.assign(
                    config_id=i,
                    preprocessor=preprocessor.name,
                    imputer=imputer.name,
                    target_scaler=target_scaler.name,
                )
                metrics = metrics.assign(
                    config_id=i,
                    preprocessor=preprocessor.name,
                    imputer=imputer.name,
                    target_scaler=target_scaler.name,
                )

                df_results = pd.concat([df_results, results])
                df_results.to_csv(f"local/eval_results/{fname}_results.csv")
                df_metrics = pd.concat([df_metrics, metrics])
                df_metrics.to_csv(f"local/eval_results/{fname}_metrics.csv")

                i += 1
