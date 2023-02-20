# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator, DummyConfidenceEstimator, ConformalKNNConfidenceEstimator, JacknifeConfidenceEstimator
from datasources.core import DefaultDataManager
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator, SupportVectorEstimator
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
from base.helper import ArcSinhScaler, DummyFeatureScaler, DummyTargetScaler, LogTargetScaler
import tqdm

if __name__ == "__main__":

    all_results = []
    num_repeats = 10
    # loads the data just like CSVDataLoader, but a selection of the data
    ft_configurations = [
        ArcSinhScaler(),
        RobustScaler(),
        PowerTransformer(standardize=False),
        StandardScaler(),
        DummyFeatureScaler(),
    ]
    tg_configurations = [
        DummyTargetScaler(),
        LogTargetScaler(),
    ]
    pbar = tqdm.tqdm(total=len(ft_configurations)*len(tg_configurations)*num_repeats)
    for i in range(num_repeats):
        dataset = DefaultDataManager().run()  # run() calls _transform()
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
        results = []  # dictionary where key=feature selection method, value = evaluation results

        for ft_scaler in ft_configurations:
            for tg_scaler in tg_configurations:
            
                start = time.time()

                ppl1 = DefaultPipeline(
                    preprocessor=IIDPreprocessor(fin_transformer=ft_scaler),
                    feature_reducer=PCAFeatureReducer(),
                    imputer=RevenueQuantileBucketImputer(),
                    scope_estimator=MiniModelArmyEstimator(),
                    ci_estimator=None,
                    scope_transformer=tg_scaler,
                ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val)


                all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 1, **ppl1.evaluation_results})
                ### EVALUATION RESULTS ###
                concatenated = pd.json_normalize(all_results)
                fname = __loader__.name.split(".")[-1]
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
                pbar.update(1)