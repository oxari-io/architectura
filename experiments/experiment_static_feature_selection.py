import argparse
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogTargetScaler
from datasources.core import FSExperimentDataLoader, get_small_datamanager_configuration, get_default_datamanager_configuration
from feature_reducers import (SelectionFeatureReducer, DropFeatureReducer, DummyFeatureReducer, FactorAnalysisFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer, PCAFeatureReducer, SparseRandProjectionFeatureReducer)
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import SupportVectorEstimator
from experiments.experiment_argument_parser import FeatureReductionExperimentCommandLineParser
import textdistance
import numpy as np

if __name__ == "__main__":
    parser = FeatureReductionExperimentCommandLineParser(
        description=
        'Experiment arguments: number of repetitions, what scopes to incorporate (-s for all 3 scopes), what file to write to (-a to append to existing file) and what feature reduction methods to compare (write -c before specifying). Defaults: 10 repititions, scope 1 only, new file, all reduction methods (DummyFeatureReducer, PCAFeatureReducer, DropFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer, SparseRandProjectionFeatureReducer, FactorAnalysisFeatureReducer).'
    )

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file

    print("num reps:", num_reps)
    print("scope: ", scope)
    print("results file: ", results_file)

    all_results = []  # dictionary where key=feature selection method, value = evaluation results
    dataset = get_default_datamanager_configuration().run()
    old_features = [
        'ft_numc_cash',
        'ft_numd_employees', 
        'ft_numc_equity',
        'ft_numc_inventories', 
        'ft_numc_market_cap', 
        'ft_numc_net_income', 
        'ft_numc_ppe',
        'ft_numc_rd_expenses', 
        'ft_numc_revenue',
        'ft_numc_roa',
        'ft_numc_roe',
        'ft_numc_stock_return',
        'ft_numc_total_assets',
        'ft_numc_total_liabilities'
    ]

    #TODO: write Monday task to correct ft_numc_rd_expenses and ft_numc_total_liabilities in frontend
    #TODO: new data misses ft_numc_market_cap, ft_numd_employees, ft_numc_stock_return
    #TODO: choose two proxies for the missing ones from the new dataset

    feature_lists = [
        old_features
    ]

    for i in range(num_reps):
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        if (scope == True):
            SPLIT_2 = bag.scope_2
            SPLIT_3 = bag.scope_3

        for feature_list in feature_lists:
            start = time.time()

            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=SelectionFeatureReducer(features=feature_list),
                imputer=RevenueQuantileBucketImputer(buckets_number=5),
                scope_estimator=SupportVectorEstimator(),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            time_elapsed_1 = time.time() - start
            start = time.time()
            all_results.append({"time": time_elapsed_1, "scope": 1, **ppl1.evaluation_results})
            
            if (scope == True):
                ppl2 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=SelectionFeatureReducer(features=feature_list),
                    imputer=RevenueQuantileBucketImputer(buckets_number=5),
                    scope_estimator=SupportVectorEstimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
                time_elapsed_2 = time.time() - start
                all_results.append({"time": time_elapsed_2, "scope": 2, **ppl2.evaluation_results})
                
                start = time.time()
                ppl3 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=SelectionFeatureReducer(features=feature_list),
                    imputer=RevenueQuantileBucketImputer(buckets_number=5),
                    scope_estimator=SupportVectorEstimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
                time_elapsed_3 = time.time() - start

                all_results.append({"time": time_elapsed_3, "scope": 3, **ppl3.evaluation_results})

            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]

            if (results_file is True):
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
            else:
                concatenated.to_csv(f'local/eval_results/{fname}.csv', header=False, mode='a')
