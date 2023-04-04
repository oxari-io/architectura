import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from base import (OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.dataset_loader import CompanyDataFilter
from base.helper import LogTargetScaler
from datasources.core import PreviousScopeFeaturesDataManager
from datasources.loaders import RegionLoader
from datastores.saver import CSVSaver, LocalDestination, MongoDestination, MongoSaver, OxariSavingManager, PickleSaver, S3Destination
from feature_reducers import DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from postprocessors import (DecisionExplainer, JumpRateExplainer, ResidualExplainer, ScopeImputerPostprocessor, ShapExplainer)
from postprocessors.missing_year_imputers import DerivativeMissingYearImputer
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from datasources.online import S3Datasource
from datasources.local import LocalDatasource

DATA_DIR = pathlib.Path('local/data')
from lar_calculator.lar_model import OxariUnboundLAR

N_TRIALS = 40
N_STARTUP_TRIALS = 10

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')

    # dataset = DefaultDataManager(scope_loader=S3ScopeLoader(), financial_loader=S3FinancialLoader(), categorical_loader=S3CategoricalLoader()).run()
    # dataset = DefaultDataManager().run()
    dataset = PreviousScopeFeaturesDataManager(
        S3Datasource(path='model-data/input/scopes_auto.csv'),
        LocalDatasource(path='model-data/input/financials_auto.csv'),
        S3Datasource(path='model-data/input/categoricals_auto.csv'),
        [RegionLoader()],
    ).set_filter(CompanyDataFilter()).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=5),
        scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=5),
        scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=5),
        scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
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
    print(eval_results)
    eval_results.T.to_csv('local/eval_results/model_pipelines.csv')
    print("Predict with Pipeline")
    # print(dp1.predict(X))
    print("Predict with Model only SCOPE1")
    print(model.predict(SPLIT_1.val.X, scope=1))

    print("\n", "Missing Year Imputation")
    data_filled = model.get_pipeline(1).preprocessor.transform(DATA)
    my_imputer = DerivativeMissingYearImputer().fit(data_filled)
    DATA_FOR_IMPUTE = my_imputer.transform(data_filled)

    print("Impute scopes with Model")
    scope_imputer = ScopeImputerPostprocessor(estimator=model).run(X=DATA_FOR_IMPUTE).evaluate()
    dataset.add_data(OxariDataManager.IMPUTED_SCOPES, scope_imputer.data, f"This data has all scopes imputed by the model on {today} at {time.localtime()}")
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
    # for idx, expl in enumerate([explainer1, explainer2, explainer3]):
    #     fig, ax = expl.plot_tree()
    #     fig.savefig(f'local/eval_results/tree_explainer{idx+1}.png', dpi=600)
    #     fig, ax = expl.plot_importances()
    #     fig.savefig(f'local/eval_results/importance_explainer{idx+1}.png')

    print("\n", "Predict ALL with Model")
    print(model.predict(SPLIT_1.val.X))

    # print("\n", "Predict ALL on Mock data")
    # print(model.predict(helper.mock_data()))

    print("\n", "Compute Confidences")
    print(model.predict(SPLIT_1.val.X, return_ci=True))

    print("\n", "DIRECT COMPARISON")
    X_new = model.predict(SPLIT_1.test.X, scope=1, return_ci=True)
    X_new["true_scope"] = SPLIT_1.test.y.values
    X_new["absolute_difference"] = np.abs(X_new["pred"] - X_new["true_scope"])
    X_new["offset_ratio"] = np.maximum(X_new["pred"], X_new["true_scope"]) / np.minimum(X_new["pred"], X_new["true_scope"])
    X_new.loc[:, SPLIT_1.test.X.columns] = SPLIT_1.test.X.values
    X_new.to_csv('local/eval_results/model_training_direct_comparison.csv')
    print(X_new)

    print("\n", "Predict LARs on Mock data")
    lar_model = OxariUnboundLAR().fit(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    lar_imputed_data = lar_model.transform(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    dataset.add_data(OxariDataManager.IMPUTED_LARS, lar_imputed_data, f"This data has all LAR values imputed by the model on {today} at {time.localtime()}")
    print(lar_imputed_data)

    tmp_pipeline = model.get_pipeline(1)

    # tmp_pipeline.feature_selector.visualize(tmp_pipeline._preprocess(X))
    ### SAVE OBJECTS ###

    all_meta_models = [
        PickleSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_model").set_object(model).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_model").set_object(model).set_datatarget(S3Destination(path="model-data/output")),
    ]

    all_lar_models = [
        PickleSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_lar").set_object(lar_model).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_lar").set_object(lar_model).set_datatarget(S3Destination(path="model-data/output")),
    ]

    df = dataset.get_data_by_name(OxariDataManager.IMPUTED_SCOPES)
    all_data_scope_imputations = [
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_scope_imputations").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_scope_imputations").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_scope_imputations").set_object(df).set_datatarget(MongoDestination(path="model-data/output")),
    ]

    df = dataset.get_data_by_name(OxariDataManager.IMPUTED_LARS)
    all_data_lar_imputations = [
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_lar_imputations").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_lar_imputations").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_lar_imputations").set_object(df).set_datatarget(MongoDestination(path="model-data/output")),
    ]

    df = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    all_data_features = [
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_companies").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_companies").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime('%d-%m-%Y')).set_name("q_companies").set_object(df).set_datatarget(MongoDestination(path="model-data/output")),
    ]

    SavingManager = OxariSavingManager(
        *all_meta_models,
        *all_lar_models,
        *all_data_scope_imputations,
        *all_data_lar_imputations,
        *all_data_features,
    )
    SavingManager.run()
