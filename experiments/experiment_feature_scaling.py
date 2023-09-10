# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd

from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator, DummyConfidenceEstimator, ConformalKNNConfidenceEstimator, JacknifeConfidenceEstimator
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from experiments.experiment_argument_parser import FeatureScalingExperimentCommandLineParser
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator, SupportVectorEstimator
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
from base.helper import ArcSinhTargetScaler, ArcSinhScaler, DummyFeatureScaler, DummyTargetScaler, LogTargetScaler
import tqdm

if __name__ == "__main__":
    parser = FeatureScalingExperimentCommandLineParser(
        description=
        "'Experiment arguments: number of repetitions, what scopes to incorporate (-s for all 3 scopes), what file to write to (-a to append to existing file). Defaults: 10 repititions, scope 1 only, new file."
    )

    all_results = []
    args = parser.parse_args()
    num_reps = args.num_reps
    scope = args.scope
    results_file = args.file
    # loads the data just like CSVDataLoader, but a selection of the data
    ft_configurations = [
        ArcSinhScaler(),
        RobustScaler(),
        PowerTransformer(),
        StandardScaler(),
        DummyFeatureScaler(),
    ]
    tg_configurations = [
        DummyTargetScaler(),
        LogTargetScaler(),
        ArcSinhTargetScaler(),
    ]

    pbar = tqdm.tqdm(total=len(ft_configurations) * len(tg_configurations) * num_reps)
    for i in range(num_reps):
        dataset = get_small_datamanager_configuration().run()  # run() calls _transform()
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
                    feature_reducer=PCAFeatureReducer(10),
                    imputer=RevenueQuantileBucketImputer(),
                    scope_estimator=MiniModelArmyEstimator(),
                    ci_estimator=None,
                    scope_transformer=tg_scaler,
                ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val)

                all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 1, **ppl1.evaluation_results})

                if (scope == True):
                    SPLIT_2 = bag.scope_2
                    ppl2 = DefaultPipeline(
                        preprocessor=IIDPreprocessor(fin_transformer=ft_scaler),
                        feature_reducer=PCAFeatureReducer(10),
                        imputer=RevenueQuantileBucketImputer(),
                        scope_estimator=MiniModelArmyEstimator(),
                        ci_estimator=None,
                        scope_transformer=tg_scaler,
                    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
                    all_results.append({"repetition": i, "time": time.time() - start, "scope": 2, **ppl2.evaluation_results})

                    SPLIT_3 = bag.scope_3
                    ppl3 = DefaultPipeline(
                        preprocessor=IIDPreprocessor(fin_transformer=ft_scaler),
                        feature_reducer=PCAFeatureReducer(10),
                        imputer=RevenueQuantileBucketImputer(),
                        scope_estimator=MiniModelArmyEstimator(),
                        ci_estimator=None,
                        scope_transformer=tg_scaler,
                    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)
                    all_results.append({"repetition": i, "time": time.time() - start, "scope": 3, **ppl3.evaluation_results})

                ### EVALUATION RESULTS ###
                concatenated = pd.json_normalize(all_results)
                fname = __loader__.name.split(".")[-1]
                concatenated.to_csv(f'local/eval_results/{fname}.csv')
                pbar.update(1)