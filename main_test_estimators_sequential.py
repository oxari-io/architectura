from pipeline.core import DefaultPipeline
from dataset_loader.csv_loader import CSVDataManager
from base import OxariDataManager, OxariPipeline
from preprocessors import BaselinePreprocessor, IIDPreprocessor, ImprovedBaselinePreprocessor
from postprocessors import ScopeImputerPostprocessor
from imputers import BaselineImputer, KMeansBucketImputer, RevenueBucketImputer, RevenueQuantileBucketImputer
from feature_reducers.core import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, LinearRegressionEstimator, BaselineEstimator, BayesianRegressionEstimator, SupportVectorEstimator, GLMEstimator
import base
from base import OxariMetaModel
import pandas as pd
# import cPickle as
import joblib as pkl
import io
import sklearn
# import multiprocessing as mp
from concurrent import futures
from itertools import product


# NOTE: IIDPreprocessor seems like a much better for most models
class Runner(object):
    def __init__(self, optimize_data, fit_data, eval_data) -> None:
        self.optimize_data = optimize_data
        self.fit_data = fit_data
        self.eval_data = eval_data

    def run(self, model: OxariPipeline):
        try:
            return model.optimise(*self.optimize_data).fit(*self.fit_data).evaluate(*self.eval_data)
        except Exception as e:
            print("Something went wrong!")
            print(e)
            return model


if __name__ == "__main__":

    dataset = CSVDataManager().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3
    model_list = [
        BayesianRegressionEstimator,
        SupportVectorEstimator,
        LinearRegressionEstimator,
        DummyEstimator,
        BaselineEstimator,
        PredictMeanEstimator,
        PredictMedianEstimator,
        GLMEstimator,
        GaussianProcessEstimator,
        MiniModelArmyEstimator,
    ]
    all_imputers = [
        RevenueQuantileBucketImputer,
        RevenueBucketImputer,
        KMeansBucketImputer,
    ]
    all_feature_reducers = [
        PCAFeatureSelector,
        # DummyFeatureReducer,
    ]
    all_preprocessors = [
        IIDPreprocessor,
        BaselinePreprocessor,
        ImprovedBaselinePreprocessor,
    ]

    all_combinations = list(product(model_list, all_preprocessors, all_imputers, all_feature_reducers, range(5)))

    all_models = [
        DefaultPipeline(
            name = f"{Model.__name__}-{idx}",
            preprocessor=Preprocessor(),
            feature_selector=FtReducer(),
            imputer=Imputer(buckets_number=5),
            scope_estimator=Model(),
        ) for Model, Preprocessor, Imputer, FtReducer, idx in all_combinations
    ]

    all_evaluations = []
    all_models_trained = []
    # TODO: how many threads? all the models in oneppol? look into this!
    optimize_data = SPLIT_1.train.X, SPLIT_1.train.y
    fit_data = SPLIT_1.train.X, SPLIT_1.train.y
    eval_data = SPLIT_1.rem.X, SPLIT_1.rem.y, SPLIT_1.val.X, SPLIT_1.val.y

    runner = Runner(optimize_data, fit_data, eval_data)
    for model in all_models:
        print(f"\n====================== MODEL: {model.name}")
        model = runner.run(model)
        all_models_trained.append(model.evaluation_results)
        eval_results = pd.json_normalize(all_models_trained)
        eval_results.to_csv('local/eval_results/results_sequential.csv')
        try:
            print(model.predict(SPLIT_1.test.X))
        except sklearn.exceptions.NotFittedError as e:
            continue
