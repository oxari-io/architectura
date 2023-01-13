from pipeline.core import DefaultPipeline, FSExperimentPipeline
from dataset_loader.csv_loader import FSExperimentDataLoader, DefaultDataManager
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver, LocalLARModelSaver, LocalDataSaver
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from postprocessors import ScopeImputerPostprocessor
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer, RevenueQuantileBucketImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector, FeatureAgglomeration, GaussRandProjection, SparseRandProjection, Factor_Analysis, Latent_Dirichlet_Allocation
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, EvenWeightMiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator, SupportVectorEstimator
from base import BaselineConfidenceEstimator, JacknifeConfidenceEstimator, DirectLossConfidenceEstimator
from base.helper import LogarithmScaler
from dataset_loader.csv_loader import DefaultDataManager
from scope_estimators import SingleBucketModelEstimator
# import base
# from base import helper
from base import OxariMetaModel
import pandas as pd
# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    for i in range(10):
        configurations = [
            BaselineConfidenceEstimator(),
            # JacknifeConfidenceEstimator(),
            DirectLossConfidenceEstimator(),
        ]
        dataset = DefaultDataManager().run()  # run() calls _transform()
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
        results = []  # dictionary where key=feature selection method, value = evaluation results
        for estimator in configurations:
            start = time.time()

            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureSelector(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=SupportVectorEstimator(),
                ci_estimator=estimator,
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
            ppl2 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureSelector(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=SupportVectorEstimator(),
                ci_estimator=estimator,
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
            ppl3 = DefaultPipeline(
                preprocessor=IIDPreprocessor(),
                feature_reducer=PCAFeatureSelector(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=SupportVectorEstimator(),
                ci_estimator=estimator,
                scope_transformer=LogarithmScaler(),
            ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 1, **ppl1.ci_estimator.evaluate(*SPLIT_1.val).evaluation_results})
            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 2, **ppl2.ci_estimator.evaluate(*SPLIT_2.val).evaluation_results})
            all_results.append({"repetition": i + 1, "time": time.time() - start, "scope": 3, **ppl3.ci_estimator.evaluate(*SPLIT_3.val).evaluation_results})
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)
            concatenated.to_csv(f'local/eval_results/{__file__}.csv')
