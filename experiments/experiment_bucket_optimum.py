# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import DummyTargetScaler, LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
from feature_reducers.core import DummyFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from imputers.core import BaselineImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from preprocessors.core import BaselinePreprocessor
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator,
                              PredictMedianEstimator,
                              SingleBucketVotingArmyEstimator)
from experiments.experiment_argument_parser import BucketingExperimentCommandLineParser



if __name__ == "__main__":

    num_reps = 10

    print("num reps:", num_reps)

    buckets = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    all_results = []
    for i in range(num_reps):
        dataset = get_small_datamanager_configuration().run()  # run() calls _transform()
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1

        for b in buckets:
            start = time.time()

            ppl1 = DefaultPipeline(
                preprocessor=BaselinePreprocessor(),
                feature_reducer=DummyFeatureReducer(),
                imputer=BaselineImputer(),
                scope_estimator=MiniModelArmyEstimator(b, n_trials=40, n_startup_trials=20),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=DummyTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val)
            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 1, **ppl1.evaluation_results})
            
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]
            
            concatenated.to_csv(f'local/eval_results/{fname}.csv')

