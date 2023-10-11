from base.constants import FEATURE_SET_VIF_UNDER_10
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers.core import DummyFeatureReducer, PCAFeatureReducer, SelectionFeatureReducer
from pipeline.core import DefaultPipeline, FSExperimentPipeline
from base import OxariDataManager
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from postprocessors import ScopeImputerPostprocessor
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer, RevenueQuantileBucketImputer
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

if __name__ == "__main__":

    all_results = []
    dataset = get_default_datamanager_configuration().run()  # run() calls _transform()
    # loads the data just like CSVDataLoader, but a selection of the data
    for i in range(10):
        configurations = [MiniModelArmyEstimator(), MLPEstimator(), LinearRegressionEstimator(), KNNEstimator(), AdaboostEstimator(), XGBEstimator(), SGDEstimator(), SupportVectorEstimator(), BaselineEstimator()]
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
        results = []  # dictionary where key=feature selection method, value = evaluation results
        for estimator in configurations:
            start = time.time()

            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=DummyFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=estimator,
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            all_results.append({"time": time.time() - start, "scope": 1, **ppl1.evaluation_results})
            ppl2 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=DummyFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=estimator,
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
            all_results.append({"time": time.time() - start, "scope": 2, **ppl2.evaluation_results})
            ppl3 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=DummyFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=estimator,
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
            all_results.append({"time": time.time() - start, "scope": 3, **ppl3.evaluation_results})
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)

            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv')
