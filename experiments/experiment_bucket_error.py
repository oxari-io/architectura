import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from base import ( OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer
from imputers.core import DummyImputer
from pipeline.core import DefaultPipeline
from postprocessors import (DecisionExplainer, JumpRateExplainer, ResidualExplainer, ScopeImputerPostprocessor, ShapExplainer)
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from preprocessors.core import NormalizedIIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from datasources.online import S3Datasource
from datasources.local import LocalDatasource
from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator
from scope_estimators.svm import FastSupportVectorEstimator
from sklearn.metrics import precision_recall_fscore_support
DATA_DIR = pathlib.Path('local/data')
from lar_calculator.lar_model import OxariUnboundLAR

N_TRIALS = 20
N_STARTUP_TRIALS = 40
# N_TRIALS = 1
# N_STARTUP_TRIALS = 1

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')

    dataset = get_small_datamanager_configuration(1).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=NormalizedIIDPreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=DummyImputer(),
        scope_estimator=EvenWeightMiniModelArmyEstimator(n_buckets=10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.test).fit_confidence(*SPLIT_1.train)

    bucket_metrics_cl = dp1.estimator.bucket_cl.bucket_metrics_
    bucket_metrics_rg = dp1.estimator.bucket_rg.bucket_specifics_["scores"]

    bucket_metrics_rg_df = pd.json_normalize([{"bucket":k, **v} for k,v in bucket_metrics_rg.items()])



    X, y = SPLIT_1.test
    X_new = dp1._preprocess(X)
    y_new = dp1._transform_scope(y)

    y_buckets = dp1.estimator.discretizer.transform(y_new)

    y_buckets_hat = dp1.estimator.bucket_cl.predict(X_new)

    raw_metrics = []

    for grp in np.unique(y_buckets):
        X_grp = X_new[y_buckets.flatten()==grp]
        y_grp = y[y_buckets.flatten()==grp]

        y_grp_hat = dp1.estimator.predict(X_grp)
        y_grp_hat_reversed = dp1._reverse_scope(y_grp_hat)

        results = dp1.estimator.bucket_rg.evaluate(y_grp, y_grp_hat_reversed)
        raw_metrics.append(results)


    raw_metrics_df = pd.json_normalize(raw_metrics)
    classification_metrics = pd.DataFrame(precision_recall_fscore_support(y_buckets, y_buckets_hat), index=["precision", "recall", "f1", "support"])

    fname = __loader__.name.split(".")[-1]
    classification_metrics.to_csv(f'local/eval_results/{fname}_classification.csv', index=False, header=True)
    raw_metrics_df.to_csv(f'local/eval_results/{fname}_regression.csv', index=False, header=True)







    # dp2 = DefaultPipeline(
    #     preprocessor=IIDPreprocessor(),
    #     feature_reducer=DummyFeatureReducer(),
    #     imputer=RevenueQuantileBucketImputer(buckets_number=5),
    #     scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
    #     ci_estimator=BaselineConfidenceEstimator(),
    #     scope_transformer=LogTargetScaler(),
    # ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    # dp3 = DefaultPipeline(
    #     preprocessor=IIDPreprocessor(),
    #     feature_reducer=DummyFeatureReducer(),
    #     imputer=RevenueQuantileBucketImputer(buckets_number=5),
    #     scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
    #     ci_estimator=BaselineConfidenceEstimator(),
    #     scope_transformer=LogTargetScaler(),
    # ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
