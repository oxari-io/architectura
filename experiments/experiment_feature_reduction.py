import argparse
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogTargetScaler
from datasources.core import FSExperimentDataLoader, get_small_datamanager_configuration
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer, FactorAnalysisFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer, PCAFeatureReducer, SparseRandProjectionFeatureReducer)
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import SupportVectorEstimator
from experiments.experiment_argument_parser import FeatureReductionExperimentCommandLineParser
import textdistance
import numpy as np


def convert_reduction_methods(reduction_methods_str_list):
    # if the reduction methods are not strings, they are already in the right format (in that case it was the default argument of parser)
    if not isinstance(reduction_methods_str_list[0], str):
        return reduction_methods_str_list

    switcher = {
        "DummyFeatureReducer": DummyFeatureReducer,
        "FeatureAgglomeration": AgglomerateFeatureReducer,
        "PCAFeatureSelector": PCAFeatureReducer,
        "DropFeatureReducer": DropFeatureReducer,
        "GaussRandProjection": GaussRandProjectionFeatureReducer,
        "SparseRandProjection": SparseRandProjectionFeatureReducer,
        "FactorAnalysis": FactorAnalysisFeatureReducer
    }

    reduction_methods = []
    for method in reduction_methods_str_list:
        m = switcher.get(method)
        if (m != None):
            reduction_methods.append(m)
        else:
            argmin = np.argmin([textdistance.damerau_levenshtein.distance(method, other) for other in switcher.keys()])
            list_of_str_methd_pairs = [l for l in switcher.items()]
            s,c = list_of_str_methd_pairs[argmin]
            print(f"Invalid method. Did you mean {s} ({c})?")
            exit()

    return reduction_methods


if __name__ == "__main__":
    parser = FeatureReductionExperimentCommandLineParser(
        description=
        'Experiment arguments: number of repetitions, what scopes to incorporate (-s for all 3 scopes), what file to write to (-a to append to existing file) and what feature reduction methods to compare (write -c before specifying). Defaults: 10 repititions, scope 1 only, new file, all reduction methods (DummyFeatureReducer, PCAFeatureReducer, DropFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer, SparseRandProjectionFeatureReducer, FactorAnalysisFeatureReducer).'
    )

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file
    reduction_methods = convert_reduction_methods(args.configurations)

    print("num reps:", num_reps)
    print("scope: ", scope)
    print("results file: ", results_file)
    print("reduction_methods: ", reduction_methods)

    all_results = []  # dictionary where key=feature selection method, value = evaluation results
    for i in range(num_reps):
        dataset = get_small_datamanager_configuration().run()
        X = dataset.get_features(OxariDataManager.ORIGINAL)
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        if (scope == True):
            SPLIT_2 = bag.scope_2
            SPLIT_3 = bag.scope_3

        # times = {}
        for selection_method in reduction_methods:
            start = time.time()

            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=selection_method(),
                imputer=RevenueQuantileBucketImputer(buckets_number=3),
                scope_estimator=SupportVectorEstimator(),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            time_elapsed_1 = time.time() - start
            start = time.time()
            all_results.append({"time": time_elapsed_1, "scope": 1, **ppl1.evaluation_results})
            if (scope == True):
                ppl2 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=selection_method(),
                    imputer=RevenueQuantileBucketImputer(buckets_number=3),
                    scope_estimator=SupportVectorEstimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
                time_elapsed_2 = time.time() - start
                start = time.time()
                ppl3 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=selection_method(),
                    imputer=RevenueQuantileBucketImputer(buckets_number=3),
                    scope_estimator=SupportVectorEstimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
                time_elapsed_3 = time.time() - start

                all_results.append({"time": time_elapsed_2, "scope": 2, **ppl2.evaluation_results})
                all_results.append({"time": time_elapsed_3, "scope": 3, **ppl3.evaluation_results})

            print(all_results)
            concatenated = pd.json_normalize(all_results)[[
                "time", "scope", "imputer", "preprocessor", "feature_selector", "scope_estimator", "test.evaluator", "test.sMAPE", "test.R2", "test.MAE", "test.RMSE", "test.MAPE"
            ]]

            fname = __loader__.name.split(".")[-1]

            if (results_file is True):
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
            else:
                concatenated.to_csv(f'local/eval_results/{fname}.csv', header=False, mode='a')
