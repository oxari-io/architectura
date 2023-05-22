# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator, DummyConfidenceEstimator, ConformalKNNConfidenceEstimator, JacknifeConfidenceEstimator
from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager, get_default_datamanager_configuration
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator, SupportVectorEstimator
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
from base.helper import ArcSinhScaler, DummyFeatureScaler, DummyTargetScaler, LogTargetScaler
import tqdm
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


if __name__ == "__main__":

    all_results = []
    num_repeats = 10
    # loads the data just like CSVDataLoader, but a selection of the data

    dataset = get_default_datamanager_configuration().run()  # run() calls _transform()
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    for i in tqdm.tqdm(range(num_repeats), desc="Overall"):
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        X, y = SPLIT_1.train
        columns = set(X.filter(regex=f"^{PreviousScopeFeaturesDataManager.PREFIX}", axis=1).columns)
        all_combinations = powerset(columns)

        pbar = tqdm.tqdm(all_combinations, desc=f"Repetition {i}")
        for to_drop in pbar:

            # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
            results = []  # dictionary where key=feature selection method, value = evaluation results

            X_train, y_train = SPLIT_1.train
            X_val, y_val = SPLIT_1.val

            X_train = X_train.drop(list(to_drop), axis=1) if len(to_drop) else X_train
            X_val = X_val.drop(list(to_drop), axis=1) if len(to_drop) else X_val

            start = time.time()
            remaining_cols = "|".join(columns.difference(to_drop))

            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
                feature_reducer=PCAFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=MiniModelArmyEstimator(),
                ci_estimator=None,
                scope_transformer=LogTargetScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.train, *SPLIT_1.val)

            all_results.append({"repetition": i + 1, "cols_used": remaining_cols, "time": time.time() - start, "scope": 1, **ppl1.evaluation_results})
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv')
