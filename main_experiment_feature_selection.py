from pipeline.core import DefaultPipeline, FSExperimentPipeline
from dataset_loader.csv_loader import CSVDataLoader, FSExperimentDataLoader
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver, LocalLARModelSaver, LocalDataSaver
from preprocessors import BaselinePreprocessor
from postprocessors import ScopeImputerPostprocessor
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector, FeatureAgglomeration, GaussRandProjection, SparseRandProjection, Factor_Analysis, Spectral_Embedding, Latent_Dirichlet_Allocation, Modified_Locally_Linear_Embedding
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator

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

    # loads the data just like CSVDataLoader, but a selection of the data
    dataset = FSExperimentDataLoader().run()
    
    # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments
    results = {} # dictionary where key=feature selection method, value = evaluation results
    for selection_method in selection_methods:
        # if (selection_method == None):
        #     selection_method = FeatureAgglomeration()
        
        dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=3),
        scope_estimator=SupportVectorEstimator(),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogarithmScaler(),
        ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)

        model = OxariMetaModel()
        postprocessor = ScopeImputerPostprocessor(estimator=model)
        model.add_pipeline(scope=3, pipeline=ppl.run_pipeline(dataset))

        X = dataset.get_data_by_name(OxariDataManager.SHORTENED)

        ### EVALUATION RESULTS ###
        print("Eval results")
        print(pd.json_normalize(model.collect_eval_results()))

        print("hello")
        # pipeline.run_pipeline(dataset) # the run_pipeline seems to be removed/altered on the main branch, so we need to find out what it has been replaced by
        # results[selection_method] = pipeline._evaluation_results # not sure how evaluation works in the code in general. I think with DefaultRegressorEvaluator, but where is that set? 

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