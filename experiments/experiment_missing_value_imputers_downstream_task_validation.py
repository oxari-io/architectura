import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from base import BaselineConfidenceEstimator, OxariDataManager
from base.common import OxariImputer
from base.constants import FEATURE_SET_VIF_UNDER_10
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
from feature_reducers.core import DummyFeatureReducer, SelectionFeatureReducer
from imputers import BaselineImputer, DummyImputer
from imputers.categorical import CategoricalStatisticsImputer, HybridCategoricalStatisticsImputer
from imputers.iterative import MVEImputer, OldOxariImputer
from imputers.kcluster_bucket import KNNBucketImputer
from imputers.other_bucket import TotalAssetsQuantileBucketImputer
from imputers.revenue_bucket import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from preprocessors.helper.custom_cat_normalizers import CountryCodeCatColumnNormalizer, IndustryNameCatColumnNormalizer, OxariCategoricalNormalizer, SectorNameCatColumnNormalizer
from scope_estimators import MiniModelArmyEstimator
from experiments.experiment_argument_parser import BucketingExperimentCommandLineParser

if __name__ == "__main__":
    all_results = []
    dataset:OxariDataManager = get_small_datamanager_configuration(1).run()
    configurations: list[OxariImputer] = [
        # AutoImputer(),
        OldOxariImputer(verbose=False),
        HybridCategoricalStatisticsImputer(),        
        DummyImputer(),
        MVEImputer(sub_estimator=MVEImputer.Strategy.DT, verbose=True),
        MVEImputer(sub_estimator=MVEImputer.Strategy.RIDGE, verbose=True),
        BaselineImputer(),
        CategoricalStatisticsImputer(reference="ft_catm_country_code"),
        CategoricalStatisticsImputer(reference="ft_catm_industry_name"),
        RevenueQuantileBucketImputer(num_buckets=11),
        TotalAssetsQuantileBucketImputer(num_buckets=11),
        KNNBucketImputer(num_buckets=9),
        # AutoImputer(AutoImputer.strategies.PMM)
    ]

    num_repeats = 10
    pbar = tqdm(total=len(configurations) * num_repeats)

    for rep in range(15):
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL, split_size_test=0.6)
        SPLIT_1 = bag.scope_1

        for imputer in configurations:
            start = time.time()
            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(cat_normalizer=OxariCategoricalNormalizer(
                    col_transformers=[SectorNameCatColumnNormalizer(),
                                      IndustryNameCatColumnNormalizer(),
                                      CountryCodeCatColumnNormalizer()])),
                feature_reducer=DummyFeatureReducer(),
                imputer=imputer,
                scope_estimator=MiniModelArmyEstimator(n_trials=20, n_startup_trials=10),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            time_elapsed_1 = time.time() - start
            all_results.append({"rep": rep, "time": time_elapsed_1, "scope": 1, **ppl1.evaluation_results})
            pbar.update(1)
            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv', header=True)
