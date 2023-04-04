import pathlib
import time
from isapi.simple import SimpleFilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from base import (OxariDataManager, OxariMetaModel, helper)
from base.common import OxariLoggerMixin, OxariPipeline
from base.constants import IMPORTANT_EVALUATION_COLUMNS
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.dataset_loader import CompanyDataFilter
from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
from datastores.saver import CSVSaver, MongoDestination, MongoSaver, PickleSaver, S3Destination
from feature_reducers import AgglomerateFeatureReducer, PCAFeatureReducer
from imputers import RevenueQuantileBucketImputer
from imputers.revenue_bucket import RevenueBucketImputer
from lar_calculator.lar_model import OxariUnboundLAR
from pipeline.core import DefaultPipeline
from postprocessors import ScopeImputerPostprocessor
from postprocessors.missing_year_imputers import DerivativeMissingYearImputer
from preprocessors import IIDPreprocessor
from scope_estimators import SupportVectorEstimator
from datastores import PartialSaver, LocalDestination, OxariSavingManager

DATA_DIR = pathlib.Path('local/data')
N_TRIALS = 5
N_STARTUP_TRIALS = 1
ENV = "t"


def compute_getting_config_works(dp1: OxariPipeline, mainlogger: OxariLoggerMixin):
    mainlogger.logger.info(f"Parameter Configuration")
    mainlogger.logger.info(f"{dp1.get_config(deep=True)}")


def computer_eval_results(model: OxariMetaModel, mainlogger: OxariLoggerMixin):
    mainlogger.logger.info(f"Evaluation results")
    eval_results = pd.json_normalize(model.collect_eval_results())
    eval_results.T.to_csv(f'local/eval_results/{ENV}_model_evalresults.csv')
    mainlogger.logger.info(f"{eval_results.loc[:, IMPORTANT_EVALUATION_COLUMNS].to_dict('records')}")


def compute_singular_prediction(model, mainlogger, scope, training_data):
    mainlogger.logger.info(f"Predict with one model")
    mainlogger.logger.info(f"Directly: {model.get_pipeline(scope).predict(training_data, scope=scope)}")
    mainlogger.logger.info(f"Predict with Model only SCOPE1 from pipeline")
    mainlogger.logger.info(f"Over Meta: {model.predict(training_data, scope=scope)}")

def compute_imputation(DATA, model, scope):
    print("\n", "Missing Year Imputation")
    data_filled = model.get_pipeline(scope).preprocessor.transform(DATA)
    my_imputer = DerivativeMissingYearImputer().fit(data_filled)
    DATA_FOR_IMPUTE = my_imputer.transform(data_filled)
    return DATA_FOR_IMPUTE

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')

    # dataset = DefaultDataManager(scope_loader=S3ScopeLoader(), financial_loader=S3FinancialLoader(), categorical_loader=S3CategoricalLoader()).run()
    # dataset = DefaultDataManager().run()
    dataset = PreviousScopeFeaturesDataManager().set_filter(CompanyDataFilter(0.01)).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    # X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # TODO: Test what happens if not all the optimise functions are called.
    # TODO: Check why scope_transformer destroys accuracy.
    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=AgglomerateFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=3),
        scope_estimator=SupportVectorEstimator(n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=SupportVectorEstimator(n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=SupportVectorEstimator(n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)

    print("HEEEEEEEEEEEEEEEERE", model.feature_names_in_)

    mainlogger = OxariLoggerMixin()

    compute_getting_config_works(dp1, mainlogger)

    ### EVALUATION RESULTS ###
    computer_eval_results(model, mainlogger)

    # print("Predict with Model only SCOPE1")
    # print(model.predict(SPLIT_1.val.X, scope=1))
    scope = 1
    training_data = SPLIT_1.val.X
    compute_singular_prediction(model, mainlogger, scope, training_data)

    DATA_FOR_IMPUTE = compute_imputation(DATA, model, scope)

    print("Impute scopes with Model")
    scope_imputer = ScopeImputerPostprocessor(estimator=model).run(X=DATA_FOR_IMPUTE).evaluate()
    dataset.add_data(OxariDataManager.IMPUTED_SCOPES, scope_imputer.data, f"This data has all scopes imputed by the model on {today} at {time.localtime()}")
    dataset.add_data(OxariDataManager.JUMP_RATES, scope_imputer.jump_rates, f"This data has jump rates per yearly transition of each company")
    dataset.add_data(OxariDataManager.JUMP_RATES_AGG, scope_imputer.jump_rates_agg, f"This data has summaries of jump-rates per company")

    scope_imputer.jump_rates.to_csv('local/eval_results/model_jump_rates_test.csv')
    scope_imputer.jump_rates_agg.to_csv('local/eval_results/model_jump_rates_agg_test.csv')

    print("\n", "Predict LARs on Mock data")
    lar_model = OxariUnboundLAR().fit(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    lar_imputed_data = lar_model.transform(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    dataset.add_data(OxariDataManager.IMPUTED_LARS, lar_imputed_data, f"This data has all LAR values imputed by the model on {today} at {time.localtime()}")
    print(lar_imputed_data)

    # print("Explain Effects of features")
    # explainer0 = ShapExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.val)
    # fig, ax = explainer0.visualize()
    # fig.savefig(f'local/eval_results/test_importance_explainer{0}.png')
    # explainer1 = ResidualExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # explainer2 = JumpRateExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # explainer3 = DecisionExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # for idx, expl in enumerate([explainer1, explainer2, explainer3]):
    #     fig, ax = expl.plot_tree()
    #     fig.savefig(f'local/eval_results/test_tree_explainer{idx+1}.png')
    #     fig, ax = expl.plot_importances()
    #     fig.savefig(f'local/eval_results/test_importance_explainer{idx+1}.png')

    # plt.show(block=True)

    print("\n", "Predict ALL with Model")
    print(model.predict(SPLIT_1.val.X))

    print("\n", "Predict ALL on Mock data")
    print(model.predict(helper.mock_data()))
    print(model.predict(helper.mock_data_dict()))

    print("\n", "Compute Confidences")
    print(model.predict(SPLIT_1.val.X, return_ci=True))

    print("\n", "DIRECT COMPARISON")
    X_new = model.predict(SPLIT_1.test.X, scope=1, return_ci=True)
    X_new["true_scope"] = SPLIT_1.test.y.values
    X_new["absolute_difference"] = np.abs(X_new["pred"] - X_new["true_scope"])
    X_new["offset_ratio"] = np.maximum(X_new["pred"], X_new["true_scope"]) / np.minimum(X_new["pred"], X_new["true_scope"])
    X_new.loc[:, SPLIT_1.test.X.columns] = SPLIT_1.test.X.values
    X_new.to_csv('local/eval_results/model_training_test.csv')
    print(X_new)

    # tmp_pipeline = model.get_pipeline(1)

    # tmp_pipeline.feature_selector.visualize(tmp_pipeline._preprocess(X))
    ### SAVE OBJECTS ###

    all_meta_models = [
        PickleSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_model").set_object(model).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_model").set_object(model).set_datatarget(S3Destination(path="model-data/output")),
    ]

    all_lar_models = [
        PickleSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_lar").set_object(lar_model).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_lar").set_object(lar_model).set_datatarget(S3Destination(path="model-data/output")),
    ]

    df = dataset.get_data_by_name(OxariDataManager.IMPUTED_SCOPES)
    all_data_scope_imputations = [
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_scope_imputations").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_scope_imputations").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_scope_imputations").set_object(df).set_datatarget(MongoDestination(path="model-data/output")),
    ]

    df = dataset.get_data_by_name(OxariDataManager.IMPUTED_LARS)
    all_data_lar_imputations = [
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_lar_imputations").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_lar_imputations").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_lar_imputations").set_object(df).set_datatarget(MongoDestination(path="model-data/output")),
    ]

    df = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    all_data_features = [
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_companies").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_companies").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime('%d-%m-%Y')).set_name("t_companies").set_object(df).set_datatarget(MongoDestination(path="model-data/output")),
    ]

    SavingManager = OxariSavingManager(
        *all_meta_models,
        *all_lar_models,
        *all_data_scope_imputations,
        *all_data_lar_imputations,
        *all_data_features,
    )
    SavingManager.run()
