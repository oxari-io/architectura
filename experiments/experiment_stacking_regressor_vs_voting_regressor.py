# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import (BaselineEstimator,
                              EvenWeightMiniModelArmyEstimator,
                              MiniModelArmyEstimator, 
                              BucketStackingArmyEstimator)
from experiments.experiment_argument_parser import StackingVsVotingExperimentCommandLineParser
# from scope_estimators.mini_model_army import AlternativeCVMiniModelArmyEstimator, CombinedMiniModelArmyEstimator


def convert_estimators(estimators_string):
    # if the estimators are not strings, they are already in the right format (in that case it was the default argument of parser)
    if not isinstance(estimators_string[0], str): 
        return estimators_string 
    
    switcher = {
        "BucketStackingArmyEstimator": BucketStackingArmyEstimator,
        "EvenWeightMiniModelArmyEstimator": EvenWeightMiniModelArmyEstimator, 
        "MiniModelArmyEstimator": MiniModelArmyEstimator, 
        "BaselineEstimator": BaselineEstimator,
    }
    
    estimators = []
    for estimator in estimators_string:
        estimators.append(switcher.get(estimator))
        
    return estimators 

if __name__ == "__main__":
    parser = StackingVsVotingExperimentCommandLineParser(description='Experiment arguments: number of repetitions, what scopes to incorporate (-s for all 3 scopes), what file to write to (-a to append to existing file) and what estimators to compare (write -c before specifying). Defaults: 10 repititions, scope 1 only, new file, estimators: BaselineEstimator, MiniModelArmyEstimator, EvenWeightMiniModelArmyEstimator, BucketStackingArmyEstimator')

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
        dataset = get_small_datamanager_configuration().run()  # run() calls _transform()
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        if (scope == True):
            SPLIT_2 = bag.scope_2
            SPLIT_3 = bag.scope_3

        for Estimator in estimators:
            start = time.time()
            bucket_specifics = {}

            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(n_components=40),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=Estimator(),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            if isinstance(ppl1.estimator, MiniModelArmyEstimator): 
                bucket_specifics = ppl1.estimator.bucket_rg.bucket_specifics_
            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 1, **ppl1.evaluation_results, **bucket_specifics})

            if (scope == True):
                ppl2 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=PCAFeatureReducer(n_components=40),
                    imputer=RevenueQuantileBucketImputer(),
                    scope_estimator=Estimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
                ppl3 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=PCAFeatureReducer(n_components=40),
                    imputer=RevenueQuantileBucketImputer(),
                    scope_estimator=Estimator(),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

                if isinstance(ppl2.estimator, MiniModelArmyEstimator): 
                    bucket_specifics = ppl2.estimator.bucket_rg.bucket_specifics_
                all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 2, **ppl2.evaluation_results, **bucket_specifics})
                
                if isinstance(ppl3.estimator, MiniModelArmyEstimator): 
                    bucket_specifics = ppl3.estimator.bucket_rg.bucket_specifics_
                all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 3, **ppl3.evaluation_results, **bucket_specifics})


            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]
            
            if (results_file is True):
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
            else: 
                concatenated.to_csv(f'local/eval_results/{fname}.csv', header = False, mode='a')

