# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator, DummyConfidenceEstimator, ConformalKNNConfidenceEstimator, JacknifeConfidenceEstimator
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from experiments.experiment_argument_parser import FeatureScalingExperimentCommandLineParser
from feature_reducers import PCAFeatureReducer
from feature_reducers.core import DummyFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from imputers.core import BaselineImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from preprocessors.core import BaselinePreprocessor, FastIndustryNormalisationBaselinePreprocessor, ImprovedBaselinePreprocessor, ImprovedIIDPreprocessor, ImprovedNormalizedIIDPreprocessor, NormalizedIIDPreprocessor
from scope_estimators import MiniModelArmyEstimator, SupportVectorEstimator
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
from base.helper import ArcSinhTargetScaler, ArcSinhScaler, DummyFeatureScaler, DummyTargetScaler, LogTargetScaler
import tqdm

if __name__ == "__main__":

    all_results = []
    num_reps = 10
    # loads the data just like CSVDataLoader, but a selection of the data
    configurations = [
        ImprovedBaselinePreprocessor(fin_transformer=PowerTransformer()),
        BaselinePreprocessor(fin_transformer=PowerTransformer()),
        ImprovedIIDPreprocessor(fin_transformer=PowerTransformer()),
        IIDPreprocessor(fin_transformer=PowerTransformer()),
        ImprovedNormalizedIIDPreprocessor(fin_transformer=PowerTransformer()),
        NormalizedIIDPreprocessor(fin_transformer=PowerTransformer()),
    ]

    pbar = tqdm.tqdm(total=len(configurations) * num_reps)
    for i in range(num_reps):
        dataset: OxariDataManager = get_small_datamanager_configuration(0.3).run()  # run() calls _transform()
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1


        # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
        results = []  # dictionary where key=feature selection method, value = evaluation results

        for preprocessor in configurations:
            start = time.time()

            ppl1 = DefaultPipeline(
                preprocessor=preprocessor,
                feature_reducer=DummyFeatureReducer(),
                imputer=BaselineImputer(),
                scope_estimator=MiniModelArmyEstimator(10, n_trials=40, n_startup_trials=20),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=DummyTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val)

            all_results.append({"repetition": i, "time": time.time() - start, "scope": 1, **ppl1.evaluation_results})

           
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv')
            pbar.update(1)