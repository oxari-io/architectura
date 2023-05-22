import time
import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager, get_default_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import (MiniModelArmyEstimator, UnderfittedClsMiniModelArmyEstimator)
from experiments.experiment_argument_parser import ClassifierPerformanceExperimentCommandLineParser
from scope_estimators.mini_model_army import MajorityClsMiniModelArmyEstimator, RandomGuessClsMiniModelArmyEstimator

if __name__ == "__main__":
    parser = ClassifierPerformanceExperimentCommandLineParser(
        description=
        "'Experiment arguments: number of repetitions, what scopes to incorporate (-s for all 3 scopes), what file to write to (-a to append to existing file). Defaults: 10 repititions, scope 1 only, new file."
    )

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file

    all_results = []
    for i in range(num_reps):
        configurations = [
            MajorityClsMiniModelArmyEstimator(),
            RandomGuessClsMiniModelArmyEstimator(),
            UnderfittedClsMiniModelArmyEstimator(),
            MiniModelArmyEstimator(),
        ]
        dataset = get_small_datamanager_configuration().run()  # run() calls _transform()
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        for estimator in configurations:
            start = time.time()
            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=estimator,
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            all_results.append({"repetition": i, "time": time.time() - start, "scope": 1, **ppl1.evaluation_results})

            if (scope == True):
                SPLIT_2 = bag.scope_2
                SPLIT_3 = bag.scope_3
                ppl2 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=PCAFeatureReducer(),
                    imputer=RevenueQuantileBucketImputer(),
                    scope_estimator=estimator,
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
                ppl3 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(),
                    feature_reducer=PCAFeatureReducer(),
                    imputer=RevenueQuantileBucketImputer(),
                    scope_estimator=estimator,
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

                all_results.append({"repetition": i, "time": time.time() - start, "scope": 2, **ppl2.evaluation_results})
                all_results.append({"repetition": i, "time": time.time() - start, "scope": 3, **ppl3.evaluation_results})

            ## EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)[[
                "scope_estimator", "repetition", "scope", "test.classifier.evaluator", "test.classifier.balanced_accuracy", "test.classifier.balanced_precision",
                "test.classifier.balanced_recall", "test.classifier.balanced_f1", "test.classifier.adj_lenient_acc", "test.classifier.adj_strict_acc", "raw.evaluator", "raw.sMAPE",
                "raw.R2", "raw.MAE", "raw.RMSE", "raw.MAPE"
            ]]
            fname = __loader__.name.split(".")[-1]

            if (results_file is True):
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
            else:
                concatenated.to_csv(f'local/eval_results/{fname}.csv', header=False, mode='a')
