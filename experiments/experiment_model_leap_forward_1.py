# pip install autoimpute
import time

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from imputers.categorical import HybridCategoricalStatisticsImputer
from imputers.core import BaselineImputer, DummyImputer
from pipeline import DefaultPipeline
from preprocessors import IIDPreprocessor
from preprocessors.core import BaselinePreprocessor
from preprocessors.helper.custom_cat_normalizers import (
    CountryCodeCatColumnNormalizer,
    LinkTransformerCatColumnNormalizer,
    OxariCategoricalNormalizer,
    IndustryNameCatColumnNormalizer,
    SectorNameCatColumnNormalizer,
)
from scope_estimators import SupportVectorEstimator, FastSupportVectorEstimator
from base.helper import ArcSinhTargetScaler, DummyTargetScaler, LogTargetScaler
from base.run_utils import (
    get_default_datamanager_configuration,
    get_remote_datamanager_configuration,
    get_small_datamanager_configuration,
)
from feature_reducers import PCAFeatureReducer, DummyFeatureReducer
from imputers import (RevenueQuantileBucketImputer)
from datasources import S3Datasource
from sklearn.preprocessing import PowerTransformer, RobustScaler, minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it

from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator, MiniModelArmyEstimator
from scope_estimators.mma.classifier import LGBMBucketClassifier, RandomForesBucketClassifier

N_TRIALS = 20
N_STARTUP_TRIALS = 40
if __name__ == "__main__":
    # TODO: Finish this experiment by adding LinearSVR
    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    dataset: OxariDataManager = get_small_datamanager_configuration().run()

    n_buckets = {
        "n_buckets_05": 5,
        "n_buckets_10": 10,
    }
    model_type = {
        "even_weighting": EvenWeightMiniModelArmyEstimator,
        "default_weighting": MiniModelArmyEstimator,
    }
    imputers = {
        "imputer_revenue_bucket_10": RevenueQuantileBucketImputer(10),
        "imputer_dummy": DummyImputer(),
        "imputer_hybrid": HybridCategoricalStatisticsImputer(),
    }

    preprocessor = {
        "preprocessor_baseline": BaselinePreprocessor,
        "preprocessor_iid": IIDPreprocessor,
    }

    scaling_ft = {
        "ft_scaling_power": PowerTransformer,
        "ft_scaling_robust": RobustScaler,
    }

    scaling_tg = {
        "tg_scaling_log": LogTargetScaler,
        "tg_scaling_arcsinh": ArcSinhTargetScaler,
    }

    bucket_classifier = {
        "clf_lgbm": LGBMBucketClassifier,
        "clf_rf": RandomForesBucketClassifier,
    }

    configurations = list(it.product(
        preprocessor.items(),
        scaling_ft.items(),
        imputers.items(),
        model_type.items(),
        n_buckets.items(),
        scaling_tg.items(),
        bucket_classifier.items(),
    ))

    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations)) as pbar:
        for i in repeats:
            bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
            SPLIT_1 = bag.scope_1
            X, Y = SPLIT_1.train
            for params in configurations:
                Preprocessor, FeatureScaler, imputer, Model, n_b, TargetScaler, BucketClassifier = params
                start_time = time.time()
                dp1 = (DefaultPipeline(
                    preprocessor=Preprocessor[1](fin_transformer=FeatureScaler[1]()),
                    feature_reducer=DummyFeatureReducer(),
                    imputer=imputer[1],
                    scope_estimator=Model(n_b[1], n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS, bucket_classifier=BucketClassifier()),
                    ci_estimator=BaselineConfidenceEstimator(),
                    scope_transformer=TargetScaler[1](),
                ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train))

                all_results.append({
                    "c_preprocessor": Preprocessor[0],
                    "c_fintransformer": FeatureScaler[0],
                    "c_imputer": imputer[0],
                    "c_model": Model[0],
                    "c_n_buckets": n_b[0],
                    "c_scope_transformer": TargetScaler[0],
                    "repetition": i,
                    "time": time.time() - start_time,
                    **dp1.evaluation_results,
                })
                concatenated = pd.json_normalize(all_results)
                fname = __loader__.name.split(".")[-1]
                pbar.update(1)
                concatenated.to_csv(f"local/eval_results/{fname}.csv")
    concatenated.to_csv(f"local/eval_results/{fname}.csv")
