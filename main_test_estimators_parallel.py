from pipeline.core import DefaultPipeline
from datasources.core import DefaultDataManager
from base import OxariDataManager, OxariPipeline
from preprocessors import BaselinePreprocessor, IIDPreprocessor, ImprovedBaselinePreprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers import KMeansBucketImputer, RevenueQuantileBucketImputer
from feature_reducers.core import DummyFeatureReducer, PCAFeatureSelector
from scope_estimators import LinearRegressionEstimator, BayesianRegressionEstimator, GLMEstimator
import pandas as pd

# import multiprocessing as mp
from concurrent import futures

def run_model(optimize_data, fit_data, eval_data, model):
    return model.optimise(*optimize_data).fit(*fit_data).evaluate(*eval_data)


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

    dataset = DefaultDataManager().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3
    model_list = [
        # REVIEWME: which sampler do the optimizer for the following estimator use?
        GLMEstimator,
        LinearRegressionEstimator,
        BayesianRegressionEstimator,
        # DummyEstimator,
        # BaselineEstimator,
        # PredictMeanEstimator,
        # PredictMedianEstimator,
        # MiniModelArmyEstimator,
        # GaussianProcessEstimator,
    ]
    all_imputers = [
        RevenueQuantileBucketImputer,
        RevenueBucketImputer,
        KMeansBucketImputer,
    ]
    all_feature_reducers = [
        PCAFeatureSelector,
        DummyFeatureReducer,
    ]
    all_preprocessors = [
        IIDPreprocessor,
        ImprovedBaselinePreprocessor,
        BaselinePreprocessor,
    ]
    all_models = [
        DefaultPipeline(
            preprocessor=Preprocessor(),
            feature_reducer=FtReducer(),
            imputer=Imputer(buckets_number=5),
            scope_estimator=Model(),
        ) for Model in model_list for Preprocessor in all_preprocessors for FtReducer in all_feature_reducers for Imputer in all_imputers
    ]

    all_evaluations = []
    all_models_trained = []
    # TODO: how many threads? all the models in oneppol? look into this!
    optimize_data = SPLIT_1.train.X, SPLIT_1.train.y
    fit_data = SPLIT_1.train.X, SPLIT_1.train.y
    eval_data = SPLIT_1.rem.X, SPLIT_1.rem.y, SPLIT_1.val.X, SPLIT_1.val.y

    runner = Runner(optimize_data, fit_data, eval_data)
    # TODO: Implement failsafe with try-except and interative csv writing
    
    with futures.ProcessPoolExecutor(8) as pool:
        for model in pool.map(runner.run, all_models):
            print(f"SAVE MODEL {model.name}")
            all_models_trained.append(model.evaluation_results)
            eval_results = pd.DataFrame(all_models_trained)
            eval_results.to_csv('local/eval_results/results_parallel.csv')

    
    # Multiprocessing also for evaluation?
    # for model in all_models_trained:
    #     all_evaluations.append(model.evaluation_results)

