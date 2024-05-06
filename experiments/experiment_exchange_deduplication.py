# pip install autoimpute
import time
from lightgbm import LGBMRegressor

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, FinancialLoader, ScopeLoader, SimpleDataFilter, StatisticalLoader
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from datasources.core import DefaultDataManager, ExchangeBasedDeduplicatedScopeDataManager
from datasources.loaders import RegionLoader
from datasources.online import CachingS3Datasource
from feature_reducers import PCAFeatureReducer
from feature_reducers.core import DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer, KNNBucketImputer, KMedianBucketImputer, BaselineImputer, RevenueBucketImputer, AutoImputer, OldOxariImputer, MVEImputer
from datasources import S3Datasource
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it

from imputers.categorical import CategoricalStatisticsImputer
from imputers.core import DummyImputer
from imputers.equilibrium_method import EquilibriumImputer, FastEquilibriumImputer
from imputers.interpolation import LinearInterpolationImputer, SplineInterpolationImputer
from imputers.kcluster_bucket import KMeansBucketImputer
from imputers.other_bucket import TotalAssetsQuantileBucketImputer, TotalLiabilitiesQuantileBucketImputer
from imputers.revenue_bucket import RevenueExponentialBucketImputer, RevenueParabolaBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors.core import NormalizedIIDPreprocessor
from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator

N_TRIALS = 40
N_STARTUP_TRIALS = 20

if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    # NOTE: I hereby define vertical interpolation and horizontal interpolation.
    # - Vertical interpolation interpolates the NA's the column independently of other columns. Usually grouped by company.
    # - Horizontal interpolation does not take any other row into account for imputation. Basically making it time-independent.
    data_fraction = 0.1
    dataset_exchange_dedup = ExchangeBasedDeduplicatedScopeDataManager(
        FinancialLoader(datasource=CachingS3Datasource(path="model-data/input/financials.csv")),
        ScopeLoader(datasource=CachingS3Datasource(path="model-data/input/scopes.csv")),
        CategoricalLoader(datasource=CachingS3Datasource(path="model-data/input/categoricals.csv")),
        StatisticalLoader(datasource=CachingS3Datasource(path="model-data/input/statisticals.csv")),
        RegionLoader(),
    ).set_filter(CompanyDataFilter(frac=data_fraction)).run()
    dataset_default = DefaultDataManager(
        FinancialLoader(datasource=CachingS3Datasource(path="model-data/input/financials.csv")),
        ScopeLoader(datasource=CachingS3Datasource(path="model-data/input/scopes.csv")),
        CategoricalLoader(datasource=CachingS3Datasource(path="model-data/input/categoricals.csv")),
        StatisticalLoader(datasource=CachingS3Datasource(path="model-data/input/statisticals.csv")),
        RegionLoader(),
    ).set_filter(CompanyDataFilter(frac=data_fraction)).run()

    configurations = [dataset_exchange_dedup, dataset_default]
    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations)) as pbar:
        for i in repeats:
            for dataset in configurations:
                bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
                SPLIT_1 = bag.scope_1
                X, Y = SPLIT_1.train
                X_new: pd.DataFrame = X.copy()

                start = time.time()

                ppl1 = DefaultPipeline(
                    preprocessor=NormalizedIIDPreprocessor(),
                    feature_reducer=DummyFeatureReducer(),
                    imputer=DummyImputer(),
                    scope_estimator=EvenWeightMiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=LogTargetScaler(),
                ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.test).fit_confidence(*SPLIT_1.train)
                all_results.append({"time": time.time() - start, "scope": 1, **ppl1.evaluation_results})

                concatenated = pd.json_normalize(all_results)
                fname = __loader__.name.split(".")[-1]
                pbar.update(1)
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
    concatenated.to_csv(f'local/eval_results/{fname}.csv')
