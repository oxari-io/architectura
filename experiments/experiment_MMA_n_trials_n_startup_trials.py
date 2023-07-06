
import time
import pandas as pd
import numpy as np

from base import BaselineConfidenceEstimator, OxariDataManager
from base.helper import LogTargetScaler
from datasources.core import get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
from imputers import BaselineImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from experiments.experiment_argument_parser import BucketingExperimentCommandLineParser

if __name__ == "__main__":
    all_results = []
    
    for data_split in range(10):
        print("data split: ", data_split)
        dataset = get_small_datamanager_configuration().run()
        DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        for rep in range(25):
            
            print("rep: ", rep, " data split: ", data_split)
            n_trials = np.random.randint(1, 50)
            n_startup_trials = np.random.randint(1, 20)
            start = time.time()
            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(n_components=6),
                imputer=BaselineImputer(),
                scope_estimator=MiniModelArmyEstimator(n_trials=n_trials, n_startup_trials=n_startup_trials),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            time_elapsed_1 = time.time() - start
            all_results.append({"time": time_elapsed_1, "scope": 1, **ppl1.evaluation_results, "n_trials": n_trials, "n_startup_trials": n_startup_trials, "data_split": data_split, "rep": rep})

            concatenated = pd.json_normalize(all_results)[["time", "scope", "imputer", "preprocessor", "feature_selector", "n_trials", "n_startup_trials", "data_split", "rep", "scope_estimator", "test.evaluator", "test.sMAPE", "test.R2", "test.MAE", "test.RMSE", "test.MAPE"]]
        
            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv', header=True)
            
