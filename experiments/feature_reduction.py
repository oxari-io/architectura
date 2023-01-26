from pipeline.core import DefaultPipeline, FSExperimentPipeline
from dataset_loader.csv_loader import FSExperimentDataLoader
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver, LocalLARModelSaver, LocalDataSaver
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from postprocessors import ScopeImputerPostprocessor
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer, RevenueQuantileBucketImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapDimensionalityReduction, MDSDimensionalitySelector, FeatureAgglomeration, GaussRandProjection, SparseRandProjection, Factor_Analysis, Latent_Dirichlet_Allocation, Spectral_Embedding 
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator, SupportVectorEstimator
from base import BaselineConfidenceEstimator
from base.helper import LogarithmScaler
from dataset_loader.csv_loader import DefaultDataManager

# import base
# from base import helper
from base import OxariMetaModel
import pandas as pd
# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import sys
import csv
import seaborn as sns
import time
import argparse

def convert_reduction_methods(reduction_methods_string):
    # if the reduction methods are not strings, they are already in the right format (in that case it was the default argument of parser)
    if not isinstance(reduction_methods_string[0], str): 
        return reduction_methods_string 
    
    switcher = {
        "DummyFeatureReducer": DummyFeatureReducer,
        "FeatureAgglomeration": FeatureAgglomeration,
        "PCAFeatureSelector": PCAFeatureSelector, 
        "DropFeatureReducer": DropFeatureReducer, 
        "GaussRandProjection": GaussRandProjection, 
        "SparseRandProjection": SparseRandProjection, 
        "Factor_Analysis": Factor_Analysis,
        "IsomapDimensionalityReduction": IsomapDimensionalityReduction, 
        "MDSDimensionalitySelector": MDSDimensionalitySelector
    }
    
    reduction_methods = []
    for method in reduction_methods_string:
        reduction_methods.append(switcher.get(method))
        
    return reduction_methods  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment arguments: number of repetitions, what scopes to incorporate (--scope-1 for scope 1 only, --scope-all for all 3 scopes), what file to write to (--append to append to existing file, --new to create new file) and what feature reduction methods to compare (use flag -methods before). Defaults: 10 repititions, --scope-1, --new, all methods.')

    # TODO: This should be --num_reps with a default of 10
    parser.add_argument('num_reps', nargs='?', default=2, type=int, help='Number of experiment repititions (default=10)')
    
    # TODO: Optimize code as boolean doesn't need two flags. Keep --scope-all
    parser.add_argument('--scope-1', dest='scope', action='store_false', help='(default) use only scope 1')
    parser.add_argument('--scope-all', dest='scope', action='store_true', help='use scopes 1, 2, 3')
    parser.set_defaults(scope=False)

    # TODO: Optimize code as boolean doesn't need two flags. [in the new argument parser class]
    parser.add_argument('--append', dest='file', action='store_false', help='append results to existing file')
    parser.add_argument('--new', dest='file', action='store_true', help='(default) store results in new file')
    parser.set_defaults(file=True)
    
    # TODO what if the naming is not exactly a class name, should this be more flexible in accepting names of reduction methods?
    # TODO: Follow CLI conventions with --methods
    parser.add_argument('-methods', dest='f_r_methods', nargs='*', type=str, default=[DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, FeatureAgglomeration, GaussRandProjection, SparseRandProjection, Factor_Analysis], help='Names of feature reduction methods to compare, use flag -methods before specifying methods')

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

        
