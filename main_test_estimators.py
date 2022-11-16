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

# import multiprocessing as mp
from concurrent import futures

if __name__ == "__main__":

    dataset = CSVDataLoader().run()
    all_models = [
        DefaultPipeline(
            scope=1,
            preprocessor=BaselinePreprocessor(),
            feature_selector=PCAFeatureSelector(),
            imputer=BaselineImputer(),
            scope_estimator=Model(),
        ) for Model in [
            # REVIEWME: which sampler do the optimizer for the following estimator use?
            LinearRegressionEstimator,
            BayesianRegressionEstimator,
            DummyEstimator,
            BaselineEstimator,
            PredictMeanEstimator,
            PredictMedianEstimator,
            MiniModelArmyEstimator,
            GaussianProcessEstimator,
        ]
    ]

    all_models_trained = []
    with futures.ProcessPoolExecutor() as pool:
        for model in all_models:
            for results in pool.map(model.run_pipeline(dataset)):
                all_models_trained.append(results)


    # Multiprocessing also for evaluation?
    all_evaluations = []
    for model in all_models_trained:
        all_evaluations.append(model.evaluation_results)

    eval_results = pd.DataFrame(all_evaluations)
    print(eval_results.to_csv('local/eval_results/junk/results.csv'))
