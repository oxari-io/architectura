import argparse
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.dataset_loader import SplitBag
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer, FactorAnalysisFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer, PCAFeatureReducer, SparseRandProjectionFeatureReducer)
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import SupportVectorEstimator
from experiments.experiment_argument_parser import ScopeEstimatorComparisonExperimentCommandLineParser
import textdistance
import numpy as np


def train_pipeline(scope: int, data: SplitBag, config, **kwargs):
    start = time.time()
    ppl = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureReducer(10),
        imputer=RevenueQuantileBucketImputer(num_buckets=5),
        scope_estimator=config(),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*data.train).fit(*data.train).evaluate(*data.rem, *data.val).fit_confidence(*data.train)
    time_elapsed = time.time() - start
    return {"time": time_elapsed, "scope": scope, **kwargs, **ppl.evaluation_results}


if __name__ == "__main__":
    parser = ScopeEstimatorComparisonExperimentCommandLineParser()

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file
    configurations = parser._convert_reduction_methods(args.configurations)

    print("num reps:", num_reps)
    print("scope: ", scope)
    print("results file: ", results_file)
    print("reduction_methods: ", configurations)

    all_results = []  # dictionary where key=feature selection method, value = evaluation results
    dataset = get_default_datamanager_configuration().run()
    for i in range(num_reps):
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

            print(all_results)
            concatenated = pd.json_normalize(all_results)

            fname = __loader__.name.split(".")[-1]

            if (results_file is True):
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
            else:
                concatenated.to_csv(f'local/eval_results/{fname}.csv', header=False, mode='a')
