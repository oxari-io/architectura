# pip install autoimpute
import time

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from pipeline import DefaultPipeline
from preprocessors import IIDPreprocessor
from preprocessors.core import BaselinePreprocessor, FastIndustryNormalisationBaselinePreprocessor
from preprocessors.helper.custom_cat_normalizers import (
    CountryCodeCatColumnNormalizer,
    LinkTransformerCatColumnNormalizer,
    OxariCategoricalNormalizer,
    IndustryNameCatColumnNormalizer,
    SectorNameCatColumnNormalizer,
)
from scope_estimators import SupportVectorEstimator, FastSupportVectorEstimator
from base.helper import LogTargetScaler
from base.run_utils import (
    get_default_datamanager_configuration,
    get_remote_datamanager_configuration,
    get_small_datamanager_configuration,
)
from feature_reducers import PCAFeatureReducer, DummyFeatureReducer
from imputers import (
    RevenueQuantileBucketImputer
)
from datasources import S3Datasource
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it

from scope_estimators.mini_model_army import MiniModelArmyEstimator
from scope_estimators.mma.classifier import LGBMBucketClassifier, RandomForesBucketClassifier, LinearSVCBucketClassifier, MLPBucketClassifier, KNNBucketClassifier, GradientBoostingBucketClassifier, GaussianNBBucketClassifier, QDABucketClassifier, SGDBucketClassifier

if __name__ == "__main__":
    # TODO: Finish this experiment by adding LinearSVR
    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    dataset = get_default_datamanager_configuration().run()

    configurations = [LGBMBucketClassifier(), RandomForesBucketClassifier(), LinearSVCBucketClassifier(), MLPBucketClassifier(), KNNBucketClassifier(), GradientBoostingBucketClassifier(), GaussianNBBucketClassifier(), QDABucketClassifier(), SGDBucketClassifier()]

    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations)) as pbar:
        for i in repeats:
            bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
            SPLIT_1 = bag.scope_1
            X, Y = SPLIT_1.train
            X_train, X_test = train_test_split(X, test_size=0.7)
            for bucket_classifier in configurations:
                dp1 = (
                    DefaultPipeline(
                        preprocessor=LinkTransformerCatColumnNormalizer(),
                        feature_reducer=DummyFeatureReducer(),
                        imputer=RevenueQuantileBucketImputer(num_buckets=10),
                        scope_estimator=MiniModelArmyEstimator(10, n_trials=40, n_startup_trials=20),
                        ci_estimator=BaselineConfidenceEstimator(),
                        scope_transformer=LogTargetScaler(),
                    )
                    .optimise(*SPLIT_1.train)
                    .fit(*SPLIT_1.train)
                    .evaluate(*SPLIT_1.rem, *SPLIT_1.val)
                    .fit_confidence(*SPLIT_1.train)
                )

                all_results.append(
                    {"configuration": bucket_classifier.__class__.__name__, "repetition": i, **dp1.evaluation_results}
                )
                concatenated = pd.json_normalize(all_results)
                fname = __loader__.name.split(".")[-1]
                pbar.update(1)
                concatenated.to_csv(f"local/eval_results/{fname}.csv")
    concatenated.to_csv(f"local/eval_results/{fname}.csv")
