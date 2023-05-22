import argparse
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.dataset_loader import SplitBag
from base.helper import ArcSinhTargetScaler, DummyTargetScaler, LogTargetScaler
from datasources.core import FSExperimentDataLoader, PreviousScopeFeaturesDataManager, get_default_datamanager_configuration
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer, FactorAnalysisFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer, PCAFeatureReducer, SparseRandProjectionFeatureReducer)
from feature_reducers.core import LDAFeatureReducer, ModifiedLocallyLinearEmbeddingFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from imputers.kcluster_bucket import KMeansBucketImputer
from imputers.revenue_bucket import RevenueBucketImputer, RevenueExponentialBucketImputer, RevenueParabolaBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from preprocessors.core import BaselinePreprocessor, ImprovedBaselinePreprocessor
from scope_estimators import SupportVectorEstimator
from experiments.experiment_argument_parser import ScopeEstimatorComparisonExperimentCommandLineParser
import textdistance
import numpy as np
from itertools import product
from scope_estimators.core import PredictMedianEstimator
from scope_estimators.linear_models import LinearRegressionEstimator
from scope_estimators.mini_model_army import MiniModelArmyEstimator
from scope_estimators.svm import FastSupportVectorEstimator
from sklearn.preprocessing import PowerTransformer

def train_pipeline(scope: int, data: SplitBag, config, **kwargs):
    _id = "_".join([obj.__class__.__name__ for obj in config])
    start = time.time()
    try:
        scope_estimator, preprocess, imputer, feature_reducer, target_scaler = config
        ppl = DefaultPipeline(
            preprocessor=preprocess,
            feature_reducer=feature_reducer,
            imputer=imputer,
            scope_estimator=scope_estimator,
            ci_estimator=BaselineConfidenceEstimator(),
            scope_transformer=target_scaler,
        ).optimise(*data.train).fit(*data.train).evaluate(*data.rem, *data.val).fit_confidence(*data.train)
        time_elapsed = time.time() - start
        return {"time": time_elapsed, "scope": scope, **kwargs, **ppl.evaluation_results, "experiment_name":_id}
    except Exception as e:
        time_elapsed = time.time() - start
        return {"time": time_elapsed, "scope": scope, **kwargs, "error":str(e), "experiment_name":_id}


if __name__ == "__main__":
    parser = ScopeEstimatorComparisonExperimentCommandLineParser()

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file


    print("num reps:", num_reps)
    print("scope: ", scope)
    print("results file: ", results_file)

    all_results = []  # dictionary where key=feature selection method, value = evaluation results
    dataset = get_default_datamanager_configuration().run()
    for i in range(num_reps):
        model_list = [
            # DummyEstimator(n_trials=5, n_startup_trials=5),
            # BaselineEstimator(n_trials=5, n_startup_trials=5),
            # PredictMeanEstimator(n_trials=5, n_startup_trials=5),
            PredictMedianEstimator(n_trials=5, n_startup_trials=5),
            # FastSupportVectorEstimator(n_trials=5, n_startup_trials=5),
            LinearRegressionEstimator(n_trials=5, n_startup_trials=5),
            MiniModelArmyEstimator(n_trials=5, n_startup_trials=5),
        ]
        all_imputers = [
            RevenueQuantileBucketImputer(7),
            RevenueBucketImputer(7),
            RevenueExponentialBucketImputer(7),
            RevenueParabolaBucketImputer(7),
            KMeansBucketImputer(7),

        ]
        all_feature_reducers = [
            PCAFeatureReducer(10),
            DummyFeatureReducer(),
            AgglomerateFeatureReducer(),
            ModifiedLocallyLinearEmbeddingFeatureReducer(10, 10),
            FactorAnalysisFeatureReducer(10),
            # LDAFeatureReducer(10), # Doesn't work with negative inputs
            # IsomapDimensionalityFeatureReducer(10), # Requires too much time to execute
            GaussRandProjectionFeatureReducer(10),
            SparseRandProjectionFeatureReducer(10),
        ]
        all_preprocessors = [
            IIDPreprocessor(),
            BaselinePreprocessor(),
            ImprovedBaselinePreprocessor(),
        ]
        all_target_scalers = [
            LogTargetScaler(),
            ArcSinhTargetScaler(),
            # PowerTransformer(),
            DummyTargetScaler(),
        ]

        configurations = list(product(model_list, all_preprocessors, all_imputers, all_feature_reducers, all_target_scalers))
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        if (scope == True):
            SPLIT_2 = bag.scope_2
            SPLIT_3 = bag.scope_3

        # times = {}
        for config in configurations:
            all_results.append(train_pipeline(1, SPLIT_1, config, repetition=i))
            if (scope == True):
                all_results.append(train_pipeline(2, SPLIT_2, config, repetition=i))
                all_results.append(train_pipeline(3, SPLIT_3, config, repetition=i))

            concatenated = pd.json_normalize(all_results)

            fname = __loader__.name.split(".")[-1]

            if (results_file is True):
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
            else:
                concatenated.to_csv(f'local/eval_results/{fname}.csv', header=False, mode='a')
