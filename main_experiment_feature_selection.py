from pipeline.core import DefaultPipeline, FSExperimentPipeline
from dataset_loader.csv_loader import FSExperimentDataLoader
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver, LocalLARModelSaver, LocalDataSaver
from preprocessors import BaselinePreprocessor
from postprocessors import ScopeImputerPostprocessor
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector, FeatureAgglomeration, GaussRandProjection, SparseRandProjection, Factor_Analysis, Latent_Dirichlet_Allocation
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator
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

if __name__ == "__main__":
    selection_methods = sys.argv[1:] # I am currently running it with the command line argument FeatureAgglomeration
    # print(selection_methods)
    selection_methods = [FeatureAgglomeration]

    # loads the data just like CSVDataLoader, but a selection of the data
    dataset = FSExperimentDataLoader().run() # run() calls _transform()
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
    results = {} # dictionary where key=feature selection method, value = evaluation results
    for selection_method in selection_methods:
        # if (selection_method == None):
        #     selection_method = FeatureAgglomeration()
        
        ppl = DefaultPipeline(
            preprocessor=BaselinePreprocessor(),
            feature_reducer=DummyFeatureReducer(),
            imputer=BaselineImputer(),
            scope_estimator=BaselineEstimator(),
            ci_estimator=BaselineConfidenceEstimator(),
            scope_transformer=LogarithmScaler(),
        ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
        

        model = OxariMetaModel()
        postprocessor = ScopeImputerPostprocessor(estimator=model)
        model.add_pipeline(scope=1, pipeline=ppl) # make scope variable later

        ### EVALUATION RESULTS ###
        print("Eval results")
        result = pd.json_normalize(model.collect_eval_results())
        pd.set_option('display.max_columns', 500)
        print(result)  
        df_smaller = pd.DataFrame()
        df_smaller = result[["imputer", "preprocessor", "feature_selector", "scope_estimator", "test.evaluator", "test.sMAPE", "test.R2", "test.MAE", "test.RMSE", "test.MAPE"]]
        print(df_smaller)



        

print("--------hello----------")
print("hi")


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