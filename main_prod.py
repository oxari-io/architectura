import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from base import (OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.dataset_loader import CategoricalLoader, FinancialLoader, ScopeLoader
from base.helper import DummyTargetScaler, LogTargetScaler
from datasources.core import PreviousScopeFeaturesDataManager, get_default_datamanager_configuration
from datasources.loaders import RegionLoader
from datastores.saver import CSVSaver, LocalDestination, MongoDestination, MongoSaver, OxariSavingManager, PickleSaver, S3Destination
from feature_reducers import DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer
from imputers.iterative import OldOxariImputer
from pipeline.core import DefaultPipeline
from postprocessors import (DecisionExplainer, JumpRateExplainer, ResidualExplainer, ScopeImputerPostprocessor, ShapExplainer)
from postprocessors.missing_year_imputers import DerivativeMissingYearImputer, SimpleMissingYearImputer
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from datasources.online import S3Datasource
from datasources.local import LocalDatasource
from lar_calculator.lar_model import OxariUnboundLAR
from pymongo import TEXT, DESCENDING, ASCENDING

from scope_estimators.svm import SupportVectorEstimator

DATA_DIR = pathlib.Path('model-data/data/input')

DATE_FORMAT = 'T%Y%m%d'

N_TRIALS = 20
N_STARTUP_TRIALS = 5
STAGE = "p_"

# TODO: Refactor experiment sections into functions (allows quick turn on and off of sections)
# TODO: Use constant STAGE to specify names for the savers (p_, q_, t_, d_)
# TODO: Use constant STAGE to specify names for intermediate savings (p_, q_, t_, d_)
# TODO: Modify saver to save all generated dataframes in OXariDataManager
# TODO: Reverse date format for saving
# TODO: Change MongoDb destiantion so that path is incorporated (Split by "database-name/collection-name")
# TODO: Remove redundant lar step from other main_* experiments too
# TODO: Introduce OxariPostprocessing piepline
# TODO: Extend CLI Runner to also include a training option
# TODO: Delete all model results and run experiments again
# TODO: Convert some of the main_*.py scripts to experiments.

if __name__ == "__main__":
    today = time.strftime(DATE_FORMAT)
    now = time.strftime('T%Y%m%d%H%M')

    dataset = get_default_datamanager_configuration().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=DummyTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=DummyTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)

    print("Parameter Configuration")
    print(dp1.get_config(deep=True))
    print(dp2.get_config(deep=True))
    print(dp3.get_config(deep=True))

    ### EVALUATION RESULTS ###
    print("Eval results")
    eval_results = pd.json_normalize(model.collect_eval_results())
    print(eval_results.T)
    eval_results.T.to_csv(f'local/prod_runs/model_pipelines_{now}.csv')
    # print("Predict with Pipeline")
    # print(dp1.predict(X))
    print("Predict with Model only SCOPE1")
    print(model.predict(SPLIT_1.val.X, scope=1))

    DATA_FOR_IMPUTE = DATA.copy()

    print("\n", "Missing Year Imputation")
    my_imputer = DerivativeMissingYearImputer().fit(DATA_FOR_IMPUTE)
    DATA_FOR_IMPUTE = my_imputer.transform(DATA_FOR_IMPUTE)

    print("Impute scopes with Model")
    scope_imputer = ScopeImputerPostprocessor(estimator=model).run(X=DATA_FOR_IMPUTE)
    dataset.add_data(OxariDataManager.IMPUTED_SCOPES, scope_imputer.data, f"This data has all scopes imputed by the model on {today} at {time.localtime()}")
    scope_imputer.data.merge(DATA, how='left', on=["key_isin", "key_year"], suffixes=[None, "_y"]).to_csv(f'local/prod_runs/model_imputations_{now}.csv')

    print('Compute jump rates')
    scope_imputer.evaluate()
    dataset.add_data(OxariDataManager.JUMP_RATES, scope_imputer.jump_rates, f"This data has jump rates per yearly transition of each company")
    dataset.add_data(OxariDataManager.JUMP_RATES_AGG, scope_imputer.jump_rates_agg, f"This data has summaries of jump-rates per company")

    scope_imputer.jump_rates.to_csv('local/eval_results/model_jump_rates.csv')
    scope_imputer.jump_rates_agg.to_csv('local/eval_results/model_jump_rates_agg.csv')

    print("\n", "Predict LARs on Mock data")
    lar_model = OxariUnboundLAR().fit(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    lar_imputed_data = lar_model.transform(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    dataset.add_data(OxariDataManager.IMPUTED_LARS, lar_imputed_data, f"This data has all LAR values imputed by the model on {today} at {time.localtime()}")
    print(lar_imputed_data)

    # print("Explain Effects of features")
    # explainer0 = ShapExplainer(model.get_pipeline(1), sample_size=100).fit(*SPLIT_1.train).explain(*SPLIT_1.val)
    # fig, ax = explainer0.visualize()
    # fig.savefig(f'local/eval_results/importance_explainer{0}.png')
    # explainer1 = ResidualExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # explainer2 = JumpRateExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # explainer3 = DecisionExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # for intervall_group, expl in enumerate([explainer1, explainer2, explainer3]):
    #     fig, ax = expl.plot_tree()
    #     fig.savefig(f'local/eval_results/tree_explainer{intervall_group+1}.png', dpi=600)
    #     fig, ax = expl.plot_importances()
    #     fig.savefig(f'local/eval_results/importance_explainer{intervall_group+1}.png')

    # print("\n", "Predict ALL with Model")
    # print(model.predict(SPLIT_1.val.X))

    # print("\n", "Predict ALL on Mock data")
    # print(model.predict(helper.mock_data()))

    # print("\n", "Compute Confidences")
    # print(model.predict(SPLIT_1.val.X, return_ci=True))

    # print("\n", "DIRECT COMPARISON")
    # X_new = model.predict(SPLIT_1.test.X, scope=1, return_ci=True)
    # X_new["true_scope"] = SPLIT_1.test.y.values
    # X_new["absolute_difference"] = np.abs(X_new["pred"] - X_new["true_scope"])
    # X_new["offset_ratio"] = np.maximum(X_new["pred"], X_new["true_scope"]) / np.minimum(X_new["pred"], X_new["true_scope"])
    # X_new.loc[:, SPLIT_1.test.X.columns] = SPLIT_1.test.X.values
    # X_new.to_csv('local/eval_results/model_training_direct_comparison.csv')
    # print(X_new)

    # tmp_pipeline = model.get_pipeline(1)

    # tmp_pipeline.feature_selector.visualize(tmp_pipeline._preprocess(X))
    ## SAVE OBJECTS ###

    all_meta_models = [
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model").set_object(model).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model").set_object(model).set_datatarget(S3Destination(path="model-data/output")),
    ]

    all_lar_models = [
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_lar").set_object(lar_model).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_lar").set_object(lar_model).set_datatarget(S3Destination(path="model-data/output")),
    ]

    df = dataset.get_data_by_name(OxariDataManager.IMPUTED_SCOPES)
    all_data_scope_imputations = [
        CSVSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".csv").set_name("p_scope_imputations").set_object(df).set_datatarget(
            LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".csv").set_name("p_scope_imputations").set_object(df).set_datatarget(
            S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime(DATE_FORMAT)).set_name("p_scope_imputations").set_object(df).set_datatarget(
            MongoDestination(path="model-data/output", index={
                "key_isin": ASCENDING,
                "key_year": ASCENDING
            })),
    ]

    df = dataset.get_data_by_name(OxariDataManager.IMPUTED_LARS)
    all_data_lar_imputations = [
        CSVSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".csv").set_name("p_lar_imputations").set_object(df).set_datatarget(
            LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".csv").set_name("p_lar_imputations").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime(DATE_FORMAT)
                              ).set_name("p_lar_imputations").set_object(df).set_datatarget(MongoDestination(path="model-data/output", index={"key_isin": ASCENDING})),
    ]

    SavingManager = OxariSavingManager(
        *all_meta_models,
        *all_lar_models,
        *all_data_scope_imputations,
        *all_data_lar_imputations,
    )
    SavingManager.run()