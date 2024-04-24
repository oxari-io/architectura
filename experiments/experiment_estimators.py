from typing import Callable
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers.core import DummyFeatureReducer, PCAFeatureReducer
from imputers.core import DummyImputer
from main_prod import N_STARTUP_TRIALS, N_TRIALS
from pipeline.core import DefaultPipeline, FSExperimentPipeline
from base import OxariDataManager
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from postprocessors import ScopeImputerPostprocessor
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer, RevenueQuantileBucketImputer
from preprocessors.core import NormalizedIIDPreprocessor
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator
from scope_estimators import SupportVectorEstimator, XGBEstimator, SGDEstimator, PLSEstimator, GaussianProcessEstimator, DummyEstimator, KNNEstimator, RNEstimator, LinearSVREstimator, BayesianRegressionEstimator, MLPEstimator
from base import BaselineConfidenceEstimator
from base.helper import LogTargetScaler
# import base
# from base import helper
from base import OxariMetaModel
import pandas as pd
# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

from scope_estimators.adaboost import AdaboostEstimator
from scope_estimators.linear_models import LinearRegressionEstimator
from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator

N_TRIALS = 20
N_STARTUP_TRIALS = 40

def spawn_model(Estimator:Callable):
    if isinstance(Estimator(), MiniModelArmyEstimator):
        return Estimator(n_buckets=10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS)
    return Estimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS)


if __name__ == "__main__":

    all_results = []
    dataset = get_default_datamanager_configuration().run()  # run() calls _transform()
    # loads the data just like CSVDataLoader, but a selection of the data
    for i in range(10):
        configurations = [EvenWeightMiniModelArmyEstimator, MLPEstimator, LinearRegressionEstimator, KNNEstimator, AdaboostEstimator, XGBEstimator, SGDEstimator, SupportVectorEstimator, BaselineEstimator]
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
        results = []  # dictionary where key=feature selection method, value = evaluation results
        for Estimator in configurations:
            start = time.time()

            ppl1 = DefaultPipeline(
                preprocessor=NormalizedIIDPreprocessor(),
                feature_reducer=DummyFeatureReducer(),
                imputer=DummyImputer(),
                scope_estimator=spawn_model(Estimator),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            all_results.append({"time": time.time() - start, "scope": 1, **ppl1.evaluation_results})

            ppl2 = DefaultPipeline(
                preprocessor=NormalizedIIDPreprocessor(),
                feature_reducer=DummyFeatureReducer(),
                imputer=DummyImputer(),
                scope_estimator=spawn_model(Estimator),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)

            all_results.append({"time": time.time() - start, "scope": 2, **ppl2.evaluation_results})
            ppl3 = DefaultPipeline(
                preprocessor=NormalizedIIDPreprocessor(),
                feature_reducer=DummyFeatureReducer(),
                imputer=DummyImputer(),
                scope_estimator=spawn_model(Estimator),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
            all_results.append({"time": time.time() - start, "scope": 3, **ppl3.evaluation_results})
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)

            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv')
