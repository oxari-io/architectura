# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator, DummyConfidenceEstimator, ConformalKNNConfidenceEstimator, JacknifeConfidenceEstimator
from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager, get_default_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator, SupportVectorEstimator

if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    for i in range(10):
        configurations = [
            ConformalKNNConfidenceEstimator,
            DummyConfidenceEstimator,
            BaselineConfidenceEstimator,
            JacknifeConfidenceEstimator,
            DirectLossConfidenceEstimator,
            PercentileOffsetConfidenceEstimator,
            MAPIEConfidenceEstimator
        ]
        dataset = get_small_datamanager_configuration().run()  # run() calls _transform()
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
        results = []  # dictionary where key=feature selection method, value = evaluation results

        ppl1 = DefaultPipeline(
            preprocessor=IIDPreprocessor(),
            feature_reducer=PCAFeatureReducer(),
            imputer=RevenueQuantileBucketImputer(),
            scope_estimator=SupportVectorEstimator(),
            ci_estimator=None,
            scope_transformer=LogTargetScaler(),
        ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val)
        ppl2 = DefaultPipeline(
            preprocessor=IIDPreprocessor(),
            feature_reducer=PCAFeatureReducer(),
            imputer=RevenueQuantileBucketImputer(),
            scope_estimator=SupportVectorEstimator(),
            ci_estimator=None,
            scope_transformer=LogTargetScaler(),
        ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val)
        ppl3 = DefaultPipeline(
            preprocessor=IIDPreprocessor(),
            feature_reducer=PCAFeatureReducer(),
            imputer=RevenueQuantileBucketImputer(),
            scope_estimator=SupportVectorEstimator(),
            ci_estimator=None,
            scope_transformer=LogTargetScaler(),
        ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val)

        for Estimator in configurations:
            start = time.time()

            ppl1.set_ci_estimator(Estimator()).fit_confidence(*SPLIT_1.train).evaluate_confidence(*SPLIT_1.val)
            ppl2.set_ci_estimator(Estimator()).fit_confidence(*SPLIT_2.train).evaluate_confidence(*SPLIT_2.val)
            ppl3.set_ci_estimator(Estimator()).fit_confidence(*SPLIT_3.train).evaluate_confidence(*SPLIT_3.val)

            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 1, **ppl1.ci_estimator.evaluation_results})
            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 2, **ppl2.ci_estimator.evaluation_results})
            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 3, **ppl3.ci_estimator.evaluation_results})
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv')