# pip install autoimpute
import time

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from pipeline import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import SupportVectorEstimator, FastSupportVectorEstimator
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer, DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer, KMeansBucketImputer, KMedianBucketImputer, BaselineImputer, RevenueBucketImputer, AutoImputer, OldOxariImputer, MVEImputer
from datasources import S3Datasource
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it
if __name__ == "__main__":
    # TODO: Finish this experiment by adding LinearSVR
    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    dataset = get_default_datamanager_configuration().run()
    configurations: list[OxariImputer] = [
        SupportVectorEstimator(), 
        FastSupportVectorEstimator(),
    ]
    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations)) as pbar:
        for i in repeats:
            bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
            SPLIT_1 = bag.scope_1
            X, Y = SPLIT_1.train
            X_numeric = X.filter(regex='^ft_num', axis=1)
            X_scaled = pd.DataFrame(minmax_scale(X_numeric, (0, 1)), columns=X_numeric.columns)
            X_train, X_test = train_test_split(X_scaled, test_size=0.7)
            for imputer in configurations:
                dp1 = DefaultPipeline(
                        preprocessor=IIDPreprocessor(),
                        feature_reducer=DummyFeatureReducer(),
                        imputer=RevenueQuantileBucketImputer(buckets_number=5),
                        scope_estimator=imputer,
                        ci_estimator=BaselineConfidenceEstimator(),
                        scope_transformer=LogTargetScaler(),
                    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
                
                all_results.append(dp1.evaluation_results)
                concatenated = pd.json_normalize(all_results)
                fname = __loader__.name.split(".")[-1]
                pbar.update(1)
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
    concatenated.to_csv(f'local/eval_results/{fname}.csv')