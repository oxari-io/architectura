import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from base import ( OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from postprocessors import (DecisionExplainer, JumpRateExplainer, ResidualExplainer, ScopeImputerPostprocessor, ShapExplainer)
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from datasources.online import S3Datasource
from datasources.local import LocalDatasource
from scope_estimators.svm import FastSupportVectorEstimator

DATA_DIR = pathlib.Path('local/data')
from lar_calculator.lar_model import OxariUnboundLAR

# N_TRIALS = 40
N_TRIALS = 1
N_STARTUP_TRIALS = 1
# N_STARTUP_TRIALS = 10

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')

    dataset = get_default_datamanager_configuration().run()
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
        imputer=RevenueQuantileBucketImputer(num_buckets=5),
        scope_estimator=FastSupportVectorEstimator(n_buckets=5, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)

    # print(f"Bucket-Specific evaluation metrics: {dp1.estimator.bucket_rg.bucket_specifics_}")
    # bucket_metrics.to_csv('local/eval_results/bucket_metrics.csv')
    bucket_metrics_cl = dp1.estimator.bucket_cl.bucket_metrics_
    bucket_metrics_rg = dp1.estimator.bucket_rg.bucket_specifics_["scores"]
    frames = [bucket_metrics_cl, bucket_metrics_rg]
    bucket_metrics =  pd.concat(frames, ignore_index=True)
    bucket_metrics.to_csv('local/eval_results/bucket_metrics.csv', index=False, header=True)

    # dp2 = DefaultPipeline(
    #     preprocessor=IIDPreprocessor(),
    #     feature_reducer=DummyFeatureReducer(),
    #     imputer=RevenueQuantileBucketImputer(buckets_number=5),
    #     scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
    #     ci_estimator=BaselineConfidenceEstimator(),
    #     scope_transformer=LogTargetScaler(),
    # ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    # dp3 = DefaultPipeline(
    #     preprocessor=IIDPreprocessor(),
    #     feature_reducer=DummyFeatureReducer(),
    #     imputer=RevenueQuantileBucketImputer(buckets_number=5),
    #     scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
    #     ci_estimator=BaselineConfidenceEstimator(),
    #     scope_transformer=LogTargetScaler(),
    # ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

