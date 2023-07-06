import numpy as np
import pandas as pd
import time

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogTargetScaler
from datasources.core import get_default_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
from imputers import BaselineImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error, mean_squared_error
from scope_estimators.mma.regressor import ExperimentOptimizer
from pmdarima.metrics import smape

if __name__ == "__main__":
    all_results = []

    dataset = get_default_datamanager_configuration().run()
    #dataset = get_small_datamanager_configuration().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)

    # loop start here
    for i in range(0, 10):
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1

        metric_names = ['smape', 'mean_squared_log_error', 'mean_squared_error']
        for metric_name in metric_names:
            if metric_name == 'smape':
                metric = smape
            if metric_name == 'mean_squared_log_error':
                metric = mean_squared_log_error
            if metric_name == 'mean_squared_error':
                metric = mean_squared_error

            start = time.time()
            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(n_components=6),
                imputer=BaselineImputer(),
                scope_estimator=MiniModelArmyEstimator(rgs_optimizer=ExperimentOptimizer(metric=metric)),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)

            predicted_values = ppl1.predict(SPLIT_1.val.X)
            residuals = SPLIT_1.val.y - predicted_values
            columns = list(SPLIT_1.val.X.columns)

            time_elapsed_1 = time.time() - start
            for residual, predicted_val, (idx, row) in zip(residuals, predicted_values, SPLIT_1.val.X.iterrows()):
                all_results.append({
                    "rep":i,
                    "time": time_elapsed_1,
                    "scope": 1,
                    **ppl1.evaluation_results, "metric": metric_name,
                    "residual": residual,
                    "y_true": residual + predicted_val,
                    "y_pred": predicted_val,
                    **row.to_dict()
                })

            concatenated = pd.json_normalize(all_results)[[
                "rep",
                "time",
                "scope",
                "imputer",
                "preprocessor",
                "feature_selector",
                "scope_estimator",
                "test.evaluator",
                "test.sMAPE",
                "test.R2",
                "test.MAE",
                "test.RMSE",
                "test.MAPE",
                "metric",
                "residual",
                "y_true",
                "y_pred",
            ] + columns]

            fname = __loader__.name.split(".")[-1]

            concatenated.to_csv(f'local/eval_results/{fname}.csv', header=True)
