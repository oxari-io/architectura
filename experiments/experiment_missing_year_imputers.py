# pip install autoimpute
import time

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager
from feature_reducers import PCAFeatureReducer
from imputers import RevenueQuantileBucketImputer, KMeansBucketImputer, KMedianBucketImputer, BaselineImputer, RevenueBucketImputer, AutoImputer, OldOxariImputer, MVEImputer
from datasources import S3Datasource
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it

from postprocessors.timeseries_interpolators import MissingYearImputer
from preprocessors.core import IIDPreprocessor
if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    dataset = DefaultDataManager(
        S3Datasource(path='model-input-data/scopes_auto.csv'),
        S3Datasource(path='model-input-data/financials_auto.csv'),
        S3Datasource(path='model-input-data/categoricals_auto.csv'),
    ).run()
    preprocessor = RevenueQuantileBucketImputer()
    configurations: list[MissingYearImputer] = [
        MissingYearImputer(),
    ]
    repeats = range(1)
    with tqdm.tqdm(total=len(repeats) * len(configurations)) as pbar:

        data = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
        data_filled = preprocessor.fit_transform(data)
        # X_features = data.filter(regex='^ft_', axis=1)
        for i in repeats:
            # X_scaled = pd.DataFrame(minmax_scale(X_numeric, (0, 1)), columns=X_numeric.columns)
            # X_train, X_test = train_test_split(data_filled, test_size=0.7)
            
            for imputer in configurations:
                # imputer = imputer
                imputer.fit(data_filled).evaluate()
                all_results.append({**imputer.evaluation_results, **imputer.get_config()})
                concatenated = pd.json_normalize(all_results)
                fname = __loader__.name.split(".")[-1]
                pbar.update(1)
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
    # concatenated.to_csv(f'local/eval_results/{fname}.csv')