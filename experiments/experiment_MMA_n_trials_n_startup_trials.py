
import time
import pandas as pd
import numpy as np

from base import BaselineConfidenceEstimator, OxariDataManager
from base.constants import FEATURE_SET_VIF_UNDER_10
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
from feature_reducers.core import SelectionFeatureReducer
from imputers import BaselineImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from experiments.experiment_argument_parser import BucketingExperimentCommandLineParser

if __name__ == "__main__":
    all_results = []
    dataset = get_default_datamanager_configuration().run()

    for data_split in range(30):
        print("data split: ", data_split)
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1

        for rep in range(10):    
            print("rep: ", rep, " data split: ", data_split)
            n_trials = np.random.randint(1, 100)
            n_startup_trials = np.random.randint(1, 50)
            start = time.time()
            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureReducer(n_components=30),
                imputer=BaselineImputer(),
                scope_estimator=MiniModelArmyEstimator(n_trials=n_trials, n_startup_trials=n_startup_trials),
                ci_estimator=BaselineConfidenceEstimator(),
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            time_elapsed_1 = time.time() - start
            all_results.append({"time": time_elapsed_1, "scope": 1, **ppl1.evaluation_results, "n_trials": n_trials, "n_startup_trials": n_startup_trials, "data_split": data_split, "rep": rep})

            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv', header=True)
            
