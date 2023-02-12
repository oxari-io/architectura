# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, JacknifeConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator
from base.helper import LogarithmScaler
from datasources.core import DefaultDataManager
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from experiments.experiment_argument_parser import ConfidenceEstimatorPerformanceExperimentCommandLineParser

def convert_confidence_estimators(estimators_string):
    # if the estimators are not strings, they are already in the right format (in that case it was the default argument of parser)
    if not isinstance(estimators_string[0], str): 
        return estimators_string 
    
    switcher = {
        "BaselineConfidenceEstimator": BaselineConfidenceEstimator,
        "JacknifeConfidenceEstimator": JacknifeConfidenceEstimator,
        "DirectLossConfidenceEstimator": DirectLossConfidenceEstimator,
        "PercentileOffsetConfidenceEstimator": PercentileOffsetConfidenceEstimator,
        "MAPIEConfidenceEstimator": MAPIEConfidenceEstimator
    }
    
    estimators = []
    for estimator in estimators_string:
        estimators.append(switcher.get(estimator))
        
    return estimators  

if __name__ == "__main__":
    parser = ConfidenceEstimatorPerformanceExperimentCommandLineParser(description='Experiment arguments: number of repetitions, what scopes to incorporate (-s for all 3 scopes), what file to write to (-a to append to existing file) and what confidence estimators to compare (write -c before specifying). Defaults: 10 repititions, scope 1 only, new file, confidence estimators: BaselineConfidenceEstimator, JacknifeConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator, MAPIEConfidenceEstimator.')

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file
    estimators = convert_confidence_estimators(args.configurations)

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

        ppl1 = DefaultPipeline(
            preprocessor=IIDPreprocessor(),
            feature_reducer=PCAFeatureReducer(),
            imputer=RevenueQuantileBucketImputer(),
            scope_estimator=MiniModelArmyEstimator(),
            ci_estimator=None,
            scope_transformer=LogarithmScaler(),
        ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val) 
        if (scope == True):     
            ppl2 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=MiniModelArmyEstimator(),
                ci_estimator=None,
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val)
            ppl3 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=MiniModelArmyEstimator(),
                ci_estimator=None,
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val)

        for Estimator in estimators:
            start = time.time()

            ppl1.set_ci_estimator(Estimator()).fit_confidence(*SPLIT_1.train).evaluate_confidence(*SPLIT_1.val)
            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 1, **ppl1.ci_estimator.evaluation_results})

            if (scope == True):
                ppl2.set_ci_estimator(Estimator()).fit_confidence(*SPLIT_2.train).evaluate_confidence(*SPLIT_2.val)
                ppl3.set_ci_estimator(Estimator()).fit_confidence(*SPLIT_3.train).evaluate_confidence(*SPLIT_3.val)
                all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 2, **ppl2.ci_estimator.evaluation_results})
                all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 3, **ppl3.ci_estimator.evaluation_results})
            
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]

            if (results_file is True):
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
            else: 
                concatenated.to_csv(f'local/eval_results/{fname}.csv', header = False, mode='a')


            