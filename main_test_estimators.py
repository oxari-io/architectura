from pipeline.core import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from base import OxariDataManager
from preprocessors import BaselinePreprocessor
from postprocessors import ScopeImputerPostprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer, KMeansBucketImputer
from feature_reducers.core import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, LinearRegressionEstimator, BaselineEstimator, BayesianRegressionEstimator
import base
from base import OxariMetaModel
import pandas as pd
# import cPickle as
import joblib as pkl
import io

# import multiprocessing as mp
from concurrent import futures

if __name__ == "__main__":

    dataset = CSVDataLoader().run()
    model_list = [
            # REVIEWME: which sampler do the optimizer for the following estimator use?
            LinearRegressionEstimator,
            BayesianRegressionEstimator,
            DummyEstimator,
            BaselineEstimator,
            PredictMeanEstimator,
            PredictMedianEstimator,
            MiniModelArmyEstimator,
            # GaussianProcessEstimator,
        ]
    all_models = [
        DefaultPipeline(
            scope=1,
            preprocessor=BaselinePreprocessor(),
            feature_selector=PCAFeatureSelector(),
            imputer=KMeansBucketImputer(buckets_number=5),
            scope_estimator=Model(),
        ) for Model in model_list
    ] 

    all_models_trained = []
    for model in all_models:
        results = model.run_pipeline(dataset)
        all_models_trained.append(results)

    # Multiprocessing also for evaluation?
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)

    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('local/eval_results/junk/results_1.csv'))

    all_models = [
        DefaultPipeline(
            scope=1,
            preprocessor=BaselinePreprocessor(),
            feature_selector=PCAFeatureSelector(),
            imputer=RevenueBucketImputer(buckets_number=5),
            scope_estimator=Model(),
        ) for Model in model_list
    ] 

    all_models_trained = []
    for model in all_models:
        results = model.run_pipeline(dataset)
        all_models_trained.append(results)

    # Multiprocessing also for evaluation?
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)

    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('local/eval_results/junk/results_2.csv'))
    
    
    all_models = [
        DefaultPipeline(
            scope=1,
            preprocessor=BaselinePreprocessor(),
            feature_selector=PCAFeatureSelector(),
            imputer=BaselineImputer(),
            scope_estimator=Model(),
        ) for Model in model_list
    ] 

    all_models_trained = []
    for model in all_models:
        results = model.run_pipeline(dataset)
        all_models_trained.append(results)

    # Multiprocessing also for evaluation?
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)

    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('local/eval_results/junk/results_3.csv')) 
    
    all_models = [
        DefaultPipeline(
            scope=1,
            preprocessor=BaselinePreprocessor(),
            feature_selector=PCAFeatureSelector(),
            imputer=KMeansBucketImputer(buckets_number=8),
            scope_estimator=Model(),
        ) for Model in model_list
    ] 

    all_models_trained = []
    for model in all_models:
        results = model.run_pipeline(dataset)
        all_models_trained.append(results)

    # Multiprocessing also for evaluation?
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)

    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('local/eval_results/junk/results_4.csv'))
    
    all_models = [
        DefaultPipeline(
            scope=1,
            preprocessor=BaselinePreprocessor(),
            feature_selector=PCAFeatureSelector(),
            imputer=KMeansBucketImputer(buckets_number=3),
            scope_estimator=Model(),
        ) for Model in model_list
    ] 

    all_models_trained = []
    for model in all_models:
        results = model.run_pipeline(dataset)
        all_models_trained.append(results)

    # Multiprocessing also for evaluation?
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)

    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('local/eval_results/junk/results_5.csv'))
    
    
    all_models = [
        DefaultPipeline(
            scope=1,
            preprocessor=BaselinePreprocessor(),
            feature_selector=PCAFeatureSelector(),
            imputer=RevenueBucketImputer(buckets_number=8),
            scope_estimator=Model(),
        ) for Model in model_list
    ] 

    all_models_trained = []
    for model in all_models:
        results = model.run_pipeline(dataset)
        all_models_trained.append(results)

    # Multiprocessing also for evaluation?
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)

    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('local/eval_results/junk/results_6.csv'))
    
    all_models = [
        DefaultPipeline(
            scope=1,
            preprocessor=BaselinePreprocessor(),
            feature_selector=PCAFeatureSelector(),
            imputer=RevenueBucketImputer(buckets_number=3),
            scope_estimator=Model(),
        ) for Model in model_list
    ] 

    all_models_trained = []
    for model in all_models:
        results = model.run_pipeline(dataset)
        all_models_trained.append(results)

    # Multiprocessing also for evaluation?
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)

    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('local/eval_results/junk/results_7.csv'))