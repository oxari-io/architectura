import pathlib
import time
import sys 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from base import (OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.constants import FEATURE_SET_VIF_UNDER_10
from base.dataset_loader import CategoricalLoader, FinancialLoader, ScopeLoader, SplitBag
from base.helper import DummyTargetScaler, LogTargetScaler
from base.run_utils import compute_jump_rates, compute_lar, create_run_report, get_small_datamanager_configuration, impute_missing_years, impute_scopes
from base.run_utils import get_default_datamanager_configuration
from datasources.loaders import RegionLoader
from datastores.saver import CSVSaver, LocalDestination, MongoDestination, MongoSaver, OxariSavingManager, PickleSaver, S3Destination
from feature_reducers import DummyFeatureReducer
from feature_reducers.core import SelectionFeatureReducer
from imputers import RevenueQuantileBucketImputer
from imputers.categorical import HybridCategoricalStatisticsImputer
from imputers.core import DummyImputer
from imputers.iterative import OldOxariImputer
from pipeline.core import DefaultPipeline
from postprocessors import (DecisionExplainer, JumpRateExplainer, ResidualExplainer, ScopeImputerPostprocessor, ShapExplainer)
from postprocessors.missing_year_imputers import DerivativeMissingYearImputer, SimpleMissingYearImputer
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from preprocessors.core import NormalizedIIDPreprocessor
from scope_estimators import MiniModelArmyEstimator, SupportVectorEstimator, FastSupportVectorEstimator
from datasources.online import S3Datasource
from datasources.local import LocalDatasource
from lar_calculator.lar_model import OxariUnboundLAR
from pymongo import TEXT, DESCENDING, ASCENDING

from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator
from scope_estimators.svm import SupportVectorEstimator

DATA_DIR = pathlib.Path('model-data/data/input')

DATE_FORMAT = 'T%Y%m%d'

N_TRIALS = 40
N_STARTUP_TRIALS = 20
STAGE = "q"

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
        preprocessor=NormalizedIIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=DummyImputer(),
        scope_estimator=FastSupportVectorEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.test).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=NormalizedIIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=DummyImputer(),
        scope_estimator=FastSupportVectorEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.test).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=NormalizedIIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=DummyImputer(),
        scope_estimator=FastSupportVectorEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.test).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)

    data = DATA.dropna(how="all").copy()
    data = data[data.filter(regex='tg_').notna().all(axis=1)]
    X = data.filter(regex='ft_', axis=1)
    Y = data.filter(regex='tg_', axis=1)
    M = data.filter(regex='key_', axis=1)
    
    bag = SplitBag(X, Y)
    model.evaluate(bag.train.X, bag.train.y, bag.test.X, bag.test.y, M)
    
    return model

def train_model_for_live_prediction(N_TRIALS, N_STARTUP_TRIALS, dataset):
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=NormalizedIIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=SelectionFeatureReducer(FEATURE_SET_VIF_UNDER_10),
        imputer=DummyImputer(),
        scope_estimator=FastSupportVectorEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.test).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=NormalizedIIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=SelectionFeatureReducer(FEATURE_SET_VIF_UNDER_10),
        imputer=DummyImputer(),
        scope_estimator=FastSupportVectorEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.test).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=NormalizedIIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=SelectionFeatureReducer(FEATURE_SET_VIF_UNDER_10),
        imputer=DummyImputer(),
        scope_estimator=FastSupportVectorEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.test).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)

    data = DATA.dropna(how="all").copy()
    data = data[data.filter(regex='tg_').notna().all(axis=1)]
    X = data.filter(regex='ft_', axis=1)
    Y = data.filter(regex='tg_', axis=1)
    M = data.filter(regex='key_', axis=1)
    
    bag = SplitBag(X, Y)
    model.evaluate(bag.train.X, bag.train.y, bag.test.X, bag.test.y, M)
    
    return model


if __name__ == "__main__":
    TODAY = time.strftime(DATE_FORMAT)
    now = time.strftime('T%Y%m%d%H%M')

    dataset = get_small_datamanager_configuration(0.1).run()
    # Scope Imputation model 
    model_si = train_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset) 
    # Live Prediciton model
    model_lp = train_model_for_live_prediction(N_TRIALS, N_STARTUP_TRIALS, dataset)

    ### EVALUATION RESULTS ###
    create_run_report(STAGE, TODAY, model_si, model_lp)

    version_info = f"python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}" 


    all_meta_models = [
        PickleSaver().set_time(TODAY).set_extension(".pkl").set_name(f"{STAGE}-model_si-{version_info}").set_object(model_si).set_datatarget(
            LocalDestination(path="model-data/output")),
        PickleSaver().set_time(TODAY).set_extension(".pkl").set_name(f"{STAGE}-model_si-{version_info}").set_object(model_si).set_datatarget(
            S3Destination(path="model-data/output")),
        PickleSaver().set_time(TODAY).set_extension(".pkl").set_name(f"{STAGE}-model-{version_info}").set_object(model_lp).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(TODAY).set_extension(".pkl").set_name(f"{STAGE}-model-{version_info}").set_object(model_lp).set_datatarget(S3Destination(path="model-data/output")),    ]

    SavingManager = OxariSavingManager(*all_meta_models, )
    SavingManager.run()