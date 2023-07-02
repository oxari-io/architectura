# pip install autoimpute
import time

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from base.dataset_loader import CompanyDataFilter
from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager, get_default_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
from imputers import RevenueQuantileBucketImputer, KMeansBucketImputer, KMedianBucketImputer, BaselineImputer, RevenueBucketImputer, AutoImputer, OldOxariImputer, MVEImputer
from datasources import S3Datasource
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it
from imputers.interpolation import LinearInterpolationImputer, SplineInterpolationImputer

from postprocessors.missing_year_imputers import QuadraticPolynomialMissingYearImputer, CubicSplineMissingYearImputer, DerivativeMissingYearImputer, DummyMissingYearImputer, SimpleMissingYearImputer
from preprocessors.core import IIDPreprocessor
if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    preprocessor = SplineInterpolationImputer()
    repeats = range(10)
    difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    configurations: list[SimpleMissingYearImputer] = [
        QuadraticPolynomialMissingYearImputer,
        DummyMissingYearImputer,
        DerivativeMissingYearImputer,
        SimpleMissingYearImputer,
        CubicSplineMissingYearImputer,
    ]
    with tqdm.tqdm(total=len(repeats) * len(configurations) * len(difficulties)) as pbar:
        for i in repeats:
            dataset = get_default_datamanager_configuration().set_filter(CompanyDataFilter(drop_single_rows=False)).run()
            data = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
            # data_filled = preprocessor.fit_transform(data)
            for Imputer in configurations:
                imputer: SimpleMissingYearImputer = Imputer()
                for dff in difficulties:
                    imputer.fit(data).evaluate(difficulty=dff)
                    all_results.append({"repetition": i, "difficulty": dff, **imputer.evaluation_results, **imputer.get_config()})
                    concatenated = pd.json_normalize(all_results)
                    fname = __loader__.name.split(".")[-1]
                    pbar.update(1)
                    concatenated.to_csv(f'local/eval_results/{fname}.csv')
    concatenated.to_csv(f'local/eval_results/{fname}.csv')