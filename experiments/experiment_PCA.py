
import time
import pandas as pd
import numpy as np

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer, DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer, BaselineImputer
from imputers.core import DummyImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor, BaselinePreprocessor
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator, SupportVectorEstimator,
                              PredictMedianEstimator,
                              SingleBucketVotingArmyEstimator)
from experiments.experiment_argument_parser import BucketingExperimentCommandLineParser, ExperimentCommandLineParser, FeatureReductionExperimentCommandLineParser
from sklearn.preprocessing import PowerTransformer

if __name__ == "__main__":
    parser = ExperimentCommandLineParser(
        description=
        'Experiment arguments: number of repetitions, what scopes to incorporate (-s for all 3 scopes), what file to write to (-a to append to existing file) and what feature reduction methods to compare (write -c before specifying). Defaults: 10 repititions, scope 1 only, new file, all reduction methods (DummyFeatureReducer, PCAFeatureReducer, DropFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer, SparseRandProjectionFeatureReducer, FactorAnalysisFeatureReducer).'
    )

    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file
    
    all_results = []

    for rep in range(num_reps):
        dataset = get_small_datamanager_configuration(0.5).run() 
        DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        if (scope == True):
            SPLIT_2 = bag.scope_2
            SPLIT_3 = bag.scope_3
        i = np.random.randint(1, 130)
        start = time.time()
        ppl1 = DefaultPipeline(
            preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
            feature_reducer=PCAFeatureReducer(n_components=i),
            imputer=DummyImputer(),
            scope_estimator=MiniModelArmyEstimator(10, n_trials=20, n_startup_trials=40),
            ci_estimator=BaselineConfidenceEstimator(),
            scope_transformer=LogTargetScaler(),
        ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
        explained_var = ppl1.feature_selector._dimensionality_reducer.explained_variance_[-1]
        time_elapsed_1 = time.time() - start
        start = time.time()
        all_results.append({"time": time_elapsed_1, "scope": 1, **ppl1.evaluation_results, "n_components": i, "repetition": rep, "variance": explained_var})
        if (scope == True):
            ppl2 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(n_components=i),
                imputer=BaselineImputer(),
                scope_estimator=MiniModelArmyEstimator(n_trials=40, n_startup_trials=20),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
            explained_var = ppl2.feature_selector._dimensionality_reducer.explained_variance_[-1]
            time_elapsed_2 = time.time() - start
            all_results.append({"time": time_elapsed_2, "scope": 2, **ppl2.evaluation_results, "n_components": i, "repetition": rep, "variance": explained_var})

            start = time.time()
            ppl3 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(n_components=i),
                imputer=BaselineImputer(),
                scope_estimator=MiniModelArmyEstimator(n_trials=40, n_startup_trials=20),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
            explained_var = ppl3.feature_selector._dimensionality_reducer.explained_variance_[-1]
            time_elapsed_3 = time.time() - start
            all_results.append({"time": time_elapsed_3, "scope": 3, **ppl3.evaluation_results, "n_components": i, "repetition": rep, "variance": explained_var})
            
        print(all_results)
        concatenated = pd.json_normalize(all_results)
        fname = __loader__.name.split(".")[-1]
        concatenated.to_csv(f'local/eval_results/{fname}.csv')

            # if (results_file is True):
            #     concatenated.to_csv(f'local/eval_results/{fname}.csv')
            # else: 
            #     concatenated.to_csv(f'local/eval_results/{fname}.csv', header = False, mode='a')

