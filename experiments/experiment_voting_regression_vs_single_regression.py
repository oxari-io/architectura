# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogarithmScaler
from datasources.core import DefaultDataManager
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator,
                              PredictMedianEstimator,
                              SingleBucketModelEstimator)
from experiments.experiment_argument_parser import VotingVsSingleExperimentCommandLineParser

def convert_estimators(estimators_string):
    # if the estimators are not strings, they are already in the right format (in that case it was the default argument of parser)
    if not isinstance(estimators_string[0], str): 
        return estimators_string 
    
    switcher = {
        "SingleBucketVotingArmyEstimator": SingleBucketModelEstimator, 
        "MiniModelArmyEstimator": MiniModelArmyEstimator, 
        "BaselineEstimator": BaselineEstimator, 
        "PredictMedianEstimator": PredictMedianEstimator
    }
    
    estimators = []
    for estimator in estimators_string:
        estimators.append(switcher.get(estimator))
        
    return estimators 

if __name__ == "__main__":
    parser = VotingVsSingleExperimentCommandLineParser(description='...')

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file
    estimators = convert_estimators(args.configurations)

    print("num reps:", num_reps)
    print("scope: ", scope)
    print("results file: ", results_file)
    print("estimators: ", estimators)

    all_results = []
    for i in range(num_reps):
        dataset = DefaultDataManager().run()  # run() calls _transform()
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        if (scope == True):
            SPLIT_2 = bag.scope_2
            SPLIT_3 = bag.scope_3

        for Estimator in estimators:
            start = time.time()

            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=Estimator(),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 1, **ppl1.evaluation_results})
            if (scope == True):
                ppl2 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=PCAFeatureReducer(),
                    imputer=RevenueQuantileBucketImputer(),
                    scope_estimator=Estimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogarithmScaler(),
                ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
                ppl3 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=PCAFeatureReducer(),
                    imputer=RevenueQuantileBucketImputer(),
                    scope_estimator=Estimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogarithmScaler(),
                ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
                
                all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 2, **ppl2.evaluation_results})
                all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 3, **ppl3.evaluation_results})

            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)[[
                "repetition",
                "time",
                "scope",
                "imputer",
                "preprocessor",
                "feature_selector",
                "scope_estimator",
                "test.evaluator",
                "test.sMAPE",
                "test.R2",
                "test.MAE",
                "test.RMSE",
                "test.MAPE",
            ]]
            
            fname = __loader__.name.split(".")[-1]
            if (results_file is True):
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
            else: 
                concatenated.to_csv(f'local/eval_results/{fname}.csv', header = False, mode='a')
