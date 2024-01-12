# pip install autoimpute
import time
from lightgbm import LGBMRegressor

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from base.dataset_loader import CompanyDataFilter, SimpleDataFilter
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
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

if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    # NOTE: I hereby define vertical interpolation and horizontal interpolation.
    # - Vertical interpolation interpolates the NA's the column independently of other columns. Usually grouped by company.
    # - Horizontal interpolation does not take any other row into account for imputation. Basically making it time-independent.
    difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dataset = get_small_datamanager_configuration(0.1).run()
    configurations: list[OxariImputer] = [
        # AutoImputer(),
        BaselineImputer(),
        DummyImputer(),
        # *[
        #     FastEquilibriumImputer(
        #         verbose=False, max_iter=50, diff_tresh=0, mims_tresh=0.0001, max_diff_increase_thresh=0.75, skip_converged_cols=True, strategy=m)
        #     for m in EquilibriumImputer.Strategy
        # ],
        *[MVEImputer(sub_estimator=m, verbose=True) for m in MVEImputer.Strategy],
        *[CategoricalStatisticsImputer(reference=ref) for ref in ["ft_catm_country_code", "ft_catm_industry_name", "ft_catm_sector_name"]],
        *[
            RevenueQuantileBucketImputer(num_buckets=11),
            TotalAssetsQuantileBucketImputer(num_buckets=11),
            TotalLiabilitiesQuantileBucketImputer(num_buckets=11)
        ],
        *[
            KNNBucketImputer(num_buckets=9),
            KMedianBucketImputer(num_buckets=13),
            KMeansBucketImputer(num_buckets=7),
        ],
        *[MVEImputer(sub_estimator=m, verbose=True) for m in [ 
            LGBMRegressor(learning_rate=0.1, n_estimators=50),
        ]],
        OldOxariImputer(verbose=True),
        # LinearInterpolationImputer(), # Vertical - Not working
        # SplineInterpolationImputer(), # Vertical - Not working
        # AutoImputer(AutoImputer.strategies.PMM)
    ]
    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations) * len(difficulties)) as pbar:
        for i in repeats:
            bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
            SPLIT_1 = bag.scope_1
            X, Y = SPLIT_1.train
            X_new:pd.DataFrame = X.copy()

            X_train, X_test = train_test_split(X_new, test_size=0.5)
            keeping_criterion_1 = (X_test.isna().mean(axis=0) < 0.5)
            keeping_criterion_2 = (X_test.isna().mean(axis=0) < 0.3)
            keep_columns_1 = X_train.loc[:, keeping_criterion_1].columns
            keep_columns_2 = X_train.loc[:, keeping_criterion_2].columns

            for imputer in configurations:
                if (i > 0) and isinstance(imputer, AutoImputer):
                    # Train this only once
                    continue
                imputer_all: OxariImputer = imputer.clone()
                imputer_1: OxariImputer = imputer.clone()
                imputer_2: OxariImputer = imputer.clone()

                imputer_all = imputer_all.fit(X_train)
                imputer_1 = imputer_1.fit(X_train[keep_columns_1])
                imputer_2 = imputer_2.fit(X_train[keep_columns_2])

                for dff in difficulties:

                    imputer_all.evaluate(X_test, p=dff)
                    all_results.append({
                        "repetition": i,
                        "difficulty": dff,
                        "mode": "realistic",
                        "num_ft": X_test.shape[1],
                        **imputer_all.evaluation_results,
                        **imputer_all.get_config()
                    })

                    imputer_1.evaluate(X_test[keep_columns_1], p=dff)
                    all_results.append({
                        "repetition": i,
                        "difficulty": dff,
                        "mode": "mid_missingness",
                        "num_ft": len(keep_columns_1),
                        **imputer_1.evaluation_results,
                        **imputer_1.get_config()
                    })

                    imputer_2.evaluate(X_test[keep_columns_2], p=dff)
                    all_results.append({
                        "repetition": i,
                        "difficulty": dff,
                        "mode": "low_missingness",
                        "num_ft": len(keep_columns_2),
                        **imputer_2.evaluation_results,
                        **imputer_2.get_config()
                    })

                    concatenated = pd.json_normalize(all_results)
                    fname = __loader__.name.split(".")[-1]
                    pbar.update(1)
                    concatenated.to_csv(f'local/eval_results/{fname}.csv')
    concatenated.to_csv(f'local/eval_results/{fname}.csv')
