from pipeline.core import DefaultPipeline, FSExperimentPipeline
from dataset_loader.csv_loader import FSExperimentDataLoader
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver, LocalLARModelSaver, LocalDataSaver
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from postprocessors import ScopeImputerPostprocessor
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer, RevenueQuantileBucketImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector, FeatureAgglomeration, GaussRandProjection, SparseRandProjection, Factor_Analysis, Latent_Dirichlet_Allocation, Spectral_Embedding 
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
        "IsomapFeatureSelector": IsomapFeatureSelector, 
        "MDSSelector": MDSSelector
    }
    
    reduction_methods = []
    for method in reduction_methods_string:
        reduction_methods.append(switcher.get(method))
        
    return reduction_methods  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process experiment arguments: what feature selection methods to compare, #reps, file to write to, what scope(s) to incorporate')

    parser.add_argument('num_reps', nargs='?', default=10, type=int, help='Number of experiment repititions')
    
    # TODO: implement True case
    # TODO: what if only one of file or scope is specified in the command line? how do you know which value belongs to which?
    parser.add_argument('scope', action='store_false', help='True if you want to include all scopes, False if you only want to include 1, default False')

    # TODO: implement False case
    parser.add_argument('results_file', action='store_true', help='True if you want to store results an empty results file, False if you want to append results to the existing file, default True')
    
    # TODO what if the naming is not exactly a class name, should this be more flexible in accepting names of reduction methods?
    parser.add_argument('f_r_methods', nargs='*', type=str, default=[DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, FeatureAgglomeration, GaussRandProjection, SparseRandProjection, Factor_Analysis], help='Names of feature reduction methods to compare')

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.results_file
    reduction_methods = args.f_r_methods

    print("num reps:", num_reps)
    print("scope: ", scope)
    print("reduction_methods: ", reduction_methods)
    print("results file: ", results_file)
    
    exit()
    all_results = [] # dictionary where key=feature selection method, value = evaluation results
    for i in range(num_reps):
        dataset = FSExperimentDataLoader().run() 
        X = dataset.get_features(OxariDataManager.ORIGINAL)
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        # times = {}
        for selection_method in reduction_methods:
            start = time.time()

            ppl = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=selection_method(),
                imputer=RevenueQuantileBucketImputer(buckets_number=3),
                scope_estimator=SupportVectorEstimator(),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            
            ### EVALUATION RESULTS ###
            print("Eval results")
            # result = pd.json_normalize(ppl._evaluation_results())
            result2 = pd.json_normalize(ppl.evaluation_results)
            
            # print(result) 
            # print(result2)  

            # all_results.append(result2)
            # end = time.time()
            # times[selection_method] = time.time()-start
            all_results.append({"time": time.time() - start, "scope": 1, **ppl.evaluation_results})

        # df_smaller = all_results[["imputer", "preprocessor", "feature_selector", "scope_estimator", "test.evaluator", "test.sMAPE", "test.R2", "test.MAE", "test.RMSE", "test.MAPE"]]
        concatenated = pd.json_normalize(all_results)[["time", "scope", "imputer", "preprocessor", "feature_selector", "scope_estimator", "test.evaluator", "test.sMAPE", "test.R2", "test.MAE", "test.RMSE", "test.MAPE"]]

        # concatenated2 = pd.concat(all_results, axis=1) #all_results now is not anymore just a list so it brings up an error when you try to concatenate it
        # dfs = [df.set_index('feature_selector') for df in results]

        # True if you want to store results an empty results file, False if you want to append results to the existing file

        if (results_file is True):
            print("true")
            concatenated.to_csv('local/eval_results/test.csv')
        else: 
            print("False")
            concatenated.to_csv('local/eval_results/test.csv', mode='a')
