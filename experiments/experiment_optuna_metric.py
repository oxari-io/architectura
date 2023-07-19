import numpy as np
import pandas as pd
import time


from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogTargetScaler
from base.metrics import mape
from datasources.core import get_default_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
from imputers import BaselineImputer
from imputers.revenue_bucket import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, mean_squared_error, median_absolute_error, r2_score
from scope_estimators.gradient_boost import XGBEstimator, XGBOptimizer
from scope_estimators.mma.regressor import ExperimentOptimizer
from pmdarima.metrics import smape

if __name__ == "__main__":
    all_results = []

    dataset = get_default_datamanager_configuration().run()
    #dataset = get_small_datamanager_configuration().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)

    # loop start here
    metric_names = {
        'mean_squared_log_error': mean_squared_log_error,
        'r2_score': r2_score,
        'mape': mean_absolute_percentage_error,
        'smape': smape,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'median_absolute_error': median_absolute_error,
    }
    for i in range(0, 20):
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1

        for metric_name, metric in metric_names.items():

            start = time.time()
            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(n_components=20),
                imputer=RevenueQuantileBucketImputer(5),
                # scope_estimator=MiniModelArmyEstimator(n_trials=20, n_startup_trials=50, rgs_optimizer=ExperimentOptimizer(metric=metric)),
                scope_estimator=XGBEstimator().set_optimizer(XGBOptimizer(n_trials=1, n_startup_trials=1, metric=metric)),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)

            indices = np.random.randint(0, len(SPLIT_1.test.X), 500)
            test_sample_X:pd.DataFrame = SPLIT_1.test.X.iloc[indices]
            test_sample_y = SPLIT_1.test.y.iloc[indices]
            predicted_values = ppl1.predict(test_sample_X)
            residuals = test_sample_y - predicted_values
            columns = list(test_sample_X.columns)

            time_elapsed_1 = time.time() - start
            for residual, predicted_val, (idx, row) in zip(residuals, predicted_values, test_sample_X.iterrows()):
                all_results.append({
                    "rep": i,
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
                "raw.sMAPE",
                "metric",
                "residual",
                "y_true",
                "y_pred",
            ] + columns]

            fname = __loader__.name.split(".")[-1]

            concatenated.to_csv(f'local/eval_results/{fname}.csv', header=True)
