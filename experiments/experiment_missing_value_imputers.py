# pip install autoimpute
import time

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from base.helper import LogarithmScaler
from datasources.core import DefaultDataManager
from feature_reducers import PCAFeatureReducer
from imputers import RevenueQuantileBucketImputer, KMeansBucketImputer, KMedianBucketImputer, BaselineImputer, RevenueBucketImputer, AutoImputer, OldOxariImputer, MVEImputer
from datasources import S3Datasource
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it
if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    dataset = DefaultDataManager(
        S3Datasource(path='model-input-data/scopes_auto.csv'),
        S3Datasource(path='model-input-data/financials_auto.csv'),
        S3Datasource(path='model-input-data/categoricals_auto.csv'),
    ).run()
    configurations: list[OxariImputer] = [
        RevenueQuantileBucketImputer(),
        KMeansBucketImputer(),
        # KMedianBucketImputer,
        BaselineImputer(),
        RevenueBucketImputer(),
        OldOxariImputer(verbose=True),
        *[MVEImputer(sub_estimator=m.value, verbose=True) for m in MVEImputer.strategies]
        # AutoImputer(),
        # AutoImputer('pmm')
    ]
    all_configs = list(it.product(range(10), configurations))
    for i, imputer in tqdm.tqdm(all_configs):
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        X, Y = SPLIT_1.train
        imputer: OxariImputer = imputer
        X_numeric = X.filter(regex='^ft_num', axis=1)
        X_scaled = pd.DataFrame(minmax_scale(X_numeric, (0, 1)), columns=X_numeric.columns)
        X_train, X_test = train_test_split(X_scaled, test_size=0.7)
        imputer.fit(X_train).evaluate(X_test)

        all_results.append(imputer.evaluation_results)
        concatenated = pd.json_normalize(all_results)
        fname = __loader__.name.split(".")[-1]
        if ((i + 1) % 5) == 0:
            concatenated.to_csv(f'local/eval_results/{fname}.csv')
    concatenated.to_csv(f'local/eval_results/{fname}.csv')