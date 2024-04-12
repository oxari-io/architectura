# pip install autoimpute
import time

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from imputers.core import BaselineImputer
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
from base.helper import DummyTargetScaler, LogTargetScaler
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

if __name__ == "__main__":
    # TODO: Finish this experiment by adding LinearSVR
    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data

    configurations = {
        "lt_default": OxariCategoricalNormalizer(
            col_transformers=[
                LinkTransformerCatColumnNormalizer(),
                CountryCodeCatColumnNormalizer(),
            ]
        ),
        "lt_default_lg": OxariCategoricalNormalizer(
            col_transformers=[
                LinkTransformerCatColumnNormalizer(
                    lt_model=LinkTransformerCatColumnNormalizer.DEFAULT_LG
                ),
                CountryCodeCatColumnNormalizer(),
            ]
        ),
        "gte_small": OxariCategoricalNormalizer(
            col_transformers=[
                LinkTransformerCatColumnNormalizer(
                    lt_model=LinkTransformerCatColumnNormalizer.GTE_SMALL
                ),
                CountryCodeCatColumnNormalizer(),
            ]
        ),
        "e5_base": OxariCategoricalNormalizer(
            col_transformers=[
                LinkTransformerCatColumnNormalizer(
                    lt_model=LinkTransformerCatColumnNormalizer.E5_BASE
                ),
                CountryCodeCatColumnNormalizer(),
            ]
        ),
        "manual": OxariCategoricalNormalizer(
            col_transformers=[
                CountryCodeCatColumnNormalizer(),
                SectorNameCatColumnNormalizer(),
                IndustryNameCatColumnNormalizer(),
            ]
        ),
    }

    repeats = range(10)
    with tqdm.tqdm(total=len(repeats) * len(configurations)) as pbar:
        for i in repeats:
            dataset:OxariDataManager = get_small_datamanager_configuration(0.25).run()
            bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
            SPLIT_1 = bag.scope_1
            X, Y = SPLIT_1.train
            for name, normalizer in configurations.items():
                start_time = time.time()
                dp1 = (
                    DefaultPipeline(
                        preprocessor=BaselinePreprocessor(cat_normalizer=normalizer),
                        feature_reducer=DummyFeatureReducer(),
                        imputer=BaselineImputer(),
                        scope_estimator=MiniModelArmyEstimator(10,n_trials=40, n_startup_trials=20),
                        ci_estimator=BaselineConfidenceEstimator(),
                        scope_transformer=DummyTargetScaler(),
                    )
                    .optimise(*SPLIT_1.train)
                    .fit(*SPLIT_1.train)
                    .evaluate(*SPLIT_1.rem, *SPLIT_1.val)
                    .fit_confidence(*SPLIT_1.train)
                )

                all_results.append(
                    {"configuration": name, "repetition": i, "time":time.time()-start_time,**dp1.evaluation_results}
                )
                concatenated = pd.json_normalize(all_results)
                fname = __loader__.name.split(".")[-1]
                pbar.update(1)
                concatenated.to_csv(f"local/eval_results/{fname}.csv")
    concatenated.to_csv(f"local/eval_results/{fname}.csv")
