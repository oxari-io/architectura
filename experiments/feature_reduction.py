import argparse
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogarithmScaler
from datasources.core import FSExperimentDataLoader
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer,
                              FactorAnalysisFeatureReducer, AgglomerateFeatureReducer,
                              GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer,
                              MDSDimensionalityFeatureReducer, PCAFeatureReducer,
                              SparseRandProjectionFeatureReducer)
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import SupportVectorEstimator
from experiment_argument_parser import FeatureReductionExperimentCommandLineParser

def convert_reduction_methods(reduction_methods_string):
    # if the reduction methods are not strings, they are already in the right format (in that case it was the default argument of parser)
    if not isinstance(reduction_methods_string[0], str): 
        return reduction_methods_string 
    
    switcher = {
        "DummyFeatureReducer": DummyFeatureReducer,
        "FeatureAgglomeration": AgglomerateFeatureReducer,
        "PCAFeatureSelector": PCAFeatureReducer, 
        "DropFeatureReducer": DropFeatureReducer, 
        "GaussRandProjection": GaussRandProjectionFeatureReducer, 
        "SparseRandProjection": SparseRandProjectionFeatureReducer, 
        "Factor_Analysis": FactorAnalysisFeatureReducer,
        "IsomapDimensionalityReduction": IsomapDimensionalityFeatureReducer, 
        "MDSDimensionalitySelector": MDSDimensionalityFeatureReducer
    }
    
    reduction_methods = []
    for method in reduction_methods_string:
        reduction_methods.append(switcher.get(method))
        
    return reduction_methods  

if __name__ == "__main__":
    parser = FeatureReductionExperimentCommandLineParser(description='Experiment arguments: number of repetitions, what scopes to incorporate (--scope-1 for scope 1 only, --scope-all for all 3 scopes), what file to write to (--append to append to existing file, --new to create new file) and what feature reduction methods to compare (use flag -methods before). Defaults: 10 repititions, --scope-1, --new, all methods.')

    # TODO chekc for illegal formats 
    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file
    reduction_methods = convert_reduction_methods(args.f_r_methods)

    print("num reps:", num_reps)
    print("scope: ", scope)
    print("results file: ", results_file)
    print("reduction_methods: ", reduction_methods)
    
    all_results = [] # dictionary where key=feature selection method, value = evaluation results
    for i in range(num_reps):
        dataset = FSExperimentDataLoader().run() 
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
                scope_transformer=LogarithmScaler(),
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
                    scope_transformer=LogarithmScaler(),
                ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
                time_elapsed_2 = time.time() - start
                start = time.time()
                ppl3 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=selection_method(),
                    imputer=RevenueQuantileBucketImputer(buckets_number=3),
                    scope_estimator=SupportVectorEstimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogarithmScaler(),
                ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
                time_elapsed_3 = time.time() - start

                all_results.append({"time": time_elapsed_2, "scope": 2, **ppl2.evaluation_results})
                all_results.append({"time": time_elapsed_3, "scope": 3, **ppl3.evaluation_results})
            
            print(all_results)
            concatenated = pd.json_normalize(all_results)[["time", "scope", "imputer", "preprocessor", "feature_selector", "scope_estimator", "test.evaluator", "test.sMAPE", "test.R2", "test.MAE", "test.RMSE", "test.MAPE"]]
            # concatenated = pd.DataFrame(concatenated)
            # print(concatenated.iloc[0])
            # print(concatenated)


            if (results_file is True):
                print("true")
                # TODO: save into file.csv with the same name as the script
                # TODO: change name of script into experiment_[SCOMETHING]
                concatenated.to_csv('local/eval_results/test.csv')
            else: 
                print("False")
                # for index, row in concatenated.iterrows():
                #     print("it goes in the loop")
                #     print(concatenated["time"].dtype)
                #     if ((concatenated["time"].str.contains("time")).any()):
                #         print("we're in here")
                #         concatenated.drop([index])
                concatenated.to_csv('local/eval_results/test.csv', header = False, mode='a')

        # df_smaller = all_results[["imputer", "preprocessor", "feature_selector", "scope_estimator", "test.evaluator", "test.sMAPE", "test.R2", "test.MAE", "test.RMSE", "test.MAPE"]]

        # concatenated2 = pd.concat(all_results, axis=1) #all_results now is not anymore just a list so it brings up an error when you try to concatenate it
        # dfs = [df.set_index('feature_selector') for df in results]

        
