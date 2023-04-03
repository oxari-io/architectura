# pip install autoimpute
import time

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from base.dataset_loader import CompanyDataFilter, SimpleDataFilter
from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager
from feature_reducers import PCAFeatureReducer
from imputers import RevenueQuantileBucketImputer, KMeansBucketImputer, KMedianBucketImputer, BaselineImputer, RevenueBucketImputer, AutoImputer, OldOxariImputer, MVEImputer
from datasources import S3Datasource
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it

from imputers.categorical import CategoricalStatisticsImputer
from imputers.interpolation import LinearInterpolationImputer, SplineInterpolationImputer
if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    # NOTE: I hereby define vertical interpolation and horizontal interpolation.
    # - Vertical interpolation interpolates the NA's the column independently of other columns. Usually grouped by company.
    # - Horizontal interpolation does not take any other row into account for imputation. Basically making it time-independent.
    difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dataset = DefaultDataManager(
        S3Datasource(path='model-input-data/scopes_auto.csv'),
        S3Datasource(path='model-input-data/financials_auto.csv'),
        S3Datasource(path='model-input-data/categoricals_auto.csv'),
    ).set_filter(CompanyDataFilter(0.1, drop_single_rows=False)).run()
    configurations: list[OxariImputer] = [
        # AutoImputer(),
        RevenueQuantileBucketImputer(),
        OldOxariImputer(verbose=True),
        KMeansBucketImputer(),
        *[MVEImputer(sub_estimator=m.value, verbose=True) for m in MVEImputer.strategies],
        # KMedianBucketImputer,
        # LinearInterpolationImputer(), # Vertical
        # SplineInterpolationImputer(), # Vertical
        CategoricalStatisticsImputer(),
        BaselineImputer(),
        RevenueBucketImputer(),
        # AutoImputer('pmm')
    ]
    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations)) as pbar:
        for i in repeats:
            bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
            SPLIT_1 = bag.scope_1
            X, Y = SPLIT_1.train
            X_new = X.copy()
            X_new[X.filter(regex='^ft_num', axis=1).columns] = minmax_scale(X.filter(regex='^ft_num', axis=1))

            X_train, X_test = train_test_split(X_new, test_size=0.7)

            for imputer in configurations:
                if (i > 0) and isinstance(imputer, AutoImputer):
                    # Train this only once
                    continue
                imputer: OxariImputer = imputer
                for dff in difficulties:
                    imputer.fit(X_train).evaluate(X_test, p=dff)
                    all_results.append({"repetition": i, "difficulty": dff, **imputer.evaluation_results, **imputer.get_config()})
                    concatenated = pd.json_normalize(all_results)
                    fname = __loader__.name.split(".")[-1]
                    pbar.update(1)
                    concatenated.to_csv(f'local/eval_results/{fname}.csv')
    concatenated.to_csv(f'local/eval_results/{fname}.csv')