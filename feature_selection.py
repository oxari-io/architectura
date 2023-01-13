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

if __name__ == "__main__":
    
    all_results = [] # dictionary where key=feature selection method, value = evaluation results
    for i in range(10):
        selection_methods = sys.argv[1:] # I am currently running it with the command line argument FeatureAgglomeration
        # print(selection_methods)
        selection_methods = [DummyFeatureReducer, FeatureAgglomeration, PCAFeatureSelector, 
        GaussRandProjection, SparseRandProjection, Factor_Analysis]

        # loads the data just like CSVDataLoader, but a selection of the data
        dataset = FSExperimentDataLoader().run() 
        X = dataset.get_features(OxariDataManager.ORIGINAL)
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
        # times = {}
        
        for selection_method in selection_methods:
            start = time.time()
            # if (selection_method == None):
            #     selection_method = FeatureAgglomeration()
            
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

        concatenated.to_csv('local/eval_results/test.csv')
        # concatenated2.to_csv('local/eval_results/test2.csv')



# print(times)


# another idea, very similar:
# if __name__ == "__main__":
    # if len(sys.argv)>1:
    #     results = {}
    #     for x in sys.argv:
    #         result = run(x)
    #         results[x] = result
              # where x is a feature selection method, and run() is the entire model run with that feature selection method


# **** DOn't inherit from class S3Datasource(Datasource)
# **** Plot results to visualise & analyze