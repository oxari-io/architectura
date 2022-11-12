from pipeline.core import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from base import OxariDataManager
from preprocessors import BaselinePreprocessor
from postprocessors import ScopeImputerPostprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers.core import BaselineImputer
from feature_reducers.core import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, LinearRegressionEstimator, BaselineEstimator, BayesianRegressionEstimator
import base
from base import OxariModel
import pandas as pd
# import cPickle as 
import joblib as pkl
import io

if __name__ == "__main__":

    dataset = CSVDataLoader().run()
    all_models = [DefaultPipeline(
        scope=1,
        preprocessor=BaselinePreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=BaselineImputer(),
        scope_estimator=Model(),
    ) for Model in [LinearRegressionEstimator, BayesianRegressionEstimator, DummyEstimator, BaselineEstimator, PredictMeanEstimator, PredictMedianEstimator, MiniModelArmyEstimator]]

    all_models_trained = []
    for model in all_models:
        all_models_trained.append(model.run_pipeline(dataset))
        
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)
        
    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('junk/results.csv'))
    
