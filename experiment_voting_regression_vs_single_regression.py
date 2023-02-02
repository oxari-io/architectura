from pipeline.core import DefaultPipeline
from base import OxariDataManager
from preprocessors import IIDPreprocessor
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from feature_reducers import PCAFeatureSelector
from scope_estimators import MiniModelArmyEstimator, BaselineEstimator
from base import BaselineConfidenceEstimator
from base.helper import LogarithmScaler
from datasources.local import DefaultDataManager
from scope_estimators import MiniModelPartyEstimator
import pandas as pd
# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    for i in range(10):
        configurations = [MiniModelArmyEstimator(), MiniModelPartyEstimator(), BaselineEstimator()]
        dataset = DefaultDataManager().run()  # run() calls _transform()
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
                feature_reducer=PCAFeatureSelector(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=estimator,
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            ppl2 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureSelector(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=estimator,
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
            ppl3 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureSelector(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=estimator,
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

            all_results.append({"time": time.time() - start, "scope": 1, **ppl1.evaluation_results})
            all_results.append({"time": time.time() - start, "scope": 2, **ppl2.evaluation_results})
            all_results.append({"time": time.time() - start, "scope": 3, **ppl3.evaluation_results})
        ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)[[
                "time", "scope", "imputer", "preprocessor", "feature_selector", "scope_estimator", "test.evaluator", "test.sMAPE", "test.R2", "test.MAE", "test.RMSE", "test.MAPE"
            ]]
            concatenated.to_csv('local/eval_results/experiment_voting_regression_vs_single_regression.csv')
