import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from base import (OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.constants import FEATURE_SET_VIF_UNDER_10
from base.dataset_loader import CategoricalLoader, FinancialLoader, ScopeLoader
from base.helper import DummyTargetScaler, LogTargetScaler
from base.run_utils import compute_jump_rates, compute_lar, impute_missing_years, impute_scopes
from base.run_utils import get_default_datamanager_configuration
from datasources.loaders import RegionLoader
from datastores.saver import CSVSaver, LocalDestination, MongoDestination, MongoSaver, OxariSavingManager, PickleSaver, S3Destination
from feature_reducers import DummyFeatureReducer
from feature_reducers.core import SelectionFeatureReducer
from imputers import RevenueQuantileBucketImputer
from imputers.iterative import OldOxariImputer
from pipeline.core import DefaultPipeline
from postprocessors import (DecisionExplainer, JumpRateExplainer, ResidualExplainer, ScopeImputerPostprocessor, ShapExplainer)
from postprocessors.missing_year_imputers import DerivativeMissingYearImputer, SimpleMissingYearImputer
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator, SupportVectorEstimator
from datasources.online import S3Datasource
from datasources.local import LocalDatasource
from lar_calculator.lar_model import OxariUnboundLAR
from pymongo import TEXT, DESCENDING, ASCENDING

from scope_estimators.svm import SupportVectorEstimator

DATA_DIR = pathlib.Path('model-data/data/input')

DATE_FORMAT = 'T%Y%m%d'

N_TRIALS = 40
N_STARTUP_TRIALS = 20
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


def train_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset):
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
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)
    return model

def train_model_for_live_prediction(N_TRIALS, N_STARTUP_TRIALS, dataset):
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=SelectionFeatureReducer(FEATURE_SET_VIF_UNDER_10),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=SelectionFeatureReducer(FEATURE_SET_VIF_UNDER_10),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=SelectionFeatureReducer(FEATURE_SET_VIF_UNDER_10),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)
    return model


if __name__ == "__main__":
    today = time.strftime(DATE_FORMAT)
    now = time.strftime('T%Y%m%d%H%M')

    dataset = get_default_datamanager_configuration().run()
    # Scope Imputation model 
    model_si = train_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset) 
    # Live Prediciton model
    model_lp = train_model_for_live_prediction(N_TRIALS, N_STARTUP_TRIALS, dataset)

    # TODO: Convert to a pytest
    # print("Parameter Configuration")
    # print(model_si.get_pipeline(1).get_config(deep=True))
    # print(model_si.get_pipeline(2).get_config(deep=True))
    # print(model_si.get_pipeline(3).get_config(deep=True))

    ### EVALUATION RESULTS ###
    print("Eval results")
    eval_results_1 = pd.json_normalize(model_si.collect_eval_results())
    eval_results_2 = pd.json_normalize(model_lp.collect_eval_results())
    pd.concat([eval_results_1, eval_results_2]).T.to_csv(f'local/prod_runs/model_pipelines_{now}.csv')

    # TODO: Convert to a pytest 
    # print("Predict with Pipeline")
    # print(dp1.predict(X))

    # TODO: Convert to a pytest
    # print("Predict with Model only SCOPE1")
    # bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    # SPLIT_1 = bag.scope_1
    # print(model.predict(SPLIT_1.val.X, scope=1))

    # TODO: Turn to a script
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

    # TODO: Convert to a pytest
    # print("\n", "Predict ALL with Model")
    # print(model.predict(SPLIT_1.val.X))

    # TODO: Convert to a pytest
    # print("\n", "Predict ALL on Mock data")
    # print(model.predict(helper.mock_data()))

    # TODO: Convert to a pytest
    # print("\n", "Compute Confidences")
    # print(model.predict(SPLIT_1.val.X, return_ci=True))

    # TODO: Convert to an analysis script
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
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model_scope_imputation").set_object(model_si).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model_scope_imputation").set_object(model_si).set_datatarget(S3Destination(path="model-data/output")),
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model").set_object(model_lp).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model").set_object(model_lp).set_datatarget(S3Destination(path="model-data/output")),
    ]

    SavingManager = OxariSavingManager(*all_meta_models, )
    SavingManager.run()