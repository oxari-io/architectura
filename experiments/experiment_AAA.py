import argparse
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.dataset_loader import SplitBag
from base.helper import ArcSinhTargetScaler, DummyTargetScaler, LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer, FactorAnalysisFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer, PCAFeatureReducer, SparseRandProjectionFeatureReducer)
from feature_reducers.core import LDAFeatureReducer, ModifiedLocallyLinearEmbeddingFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from imputers.categorical import CategoricalStatisticsImputer, HybridCategoricalStatisticsImputer
from imputers.core import DummyImputer
from imputers.kcluster_bucket import KNNBucketImputer
from imputers.revenue_bucket import RevenueBucketImputer
from main_prod import N_STARTUP_TRIALS
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from preprocessors.core import BaselinePreprocessor, ImprovedBaselinePreprocessor
from scope_estimators import SupportVectorEstimator
from experiments.experiment_argument_parser import ScopeEstimatorComparisonExperimentCommandLineParser
import textdistance
import numpy as np
from itertools import product
from scope_estimators.core import BaselineEstimator, DummyEstimator, PredictMeanEstimator, PredictMedianEstimator
from scope_estimators.linear_models import LinearRegressionEstimator
from scope_estimators.mini_model_army import MiniModelArmyEstimator
from scope_estimators.svm import FastSupportVectorEstimator
from sklearn.preprocessing import PowerTransformer
from scope_estimators.gradient_boost import XGBEstimator
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

N_STARTUP_TRIALS = 40
N_TRIALS = 40

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
    dataset = get_small_datamanager_configuration(0.5).run()
    for i in range(num_reps):
        model_list = [
            BaselineEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
            PredictMeanEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
            PredictMedianEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
            FastSupportVectorEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
            LinearRegressionEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
            XGBEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
            MiniModelArmyEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
            DummyEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ]
        all_imputers = [
            RevenueQuantileBucketImputer(10),
            # RevenueBucketImputer(7),
            # KNNBucketImputer(10),
            DummyImputer(),
            HybridCategoricalStatisticsImputer(),
            CategoricalStatisticsImputer(reference="ft_catm_country_code")

        ]
        all_feature_reducers = [
            PCAFeatureReducer(10),
            DummyFeatureReducer(),
            # AgglomerateFeatureReducer(),
            # ModifiedLocallyLinearEmbeddingFeatureReducer(10, 10),
            # FactorAnalysisFeatureReducer(10),
            # LDAFeatureReducer(10), # Doesn't work with negative inputs
            # IsomapDimensionalityFeatureReducer(10), # Requires too much time to execute
            # GaussRandProjectionFeatureReducer(10),
            # SparseRandProjectionFeatureReducer(10),
        ]
        all_preprocessors = [
            IIDPreprocessor(fin_transformer=PowerTransformer()),
            BaselinePreprocessor(fin_transformer=PowerTransformer()),
            # ImprovedBaselinePreprocessor(),
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
