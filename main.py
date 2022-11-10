from pipeline.baseline import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from scope_estimators.mini_model_army import MiniModelArmyEstimator
from preprocessors.baseline import BaselinePreprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers.baseline import DummyImputer
from feature_reducers.baseline import DummyFeatureSelector, PCAFeatureSelector, DropFeatureSelector
from scope_estimators.mma.classifier import ClassifierOptimizer
from scope_estimators.gaussian_process import GaussianProcessEstimator
import base
from base import OxariModel

if __name__ == "__main__":

    dataset=CSVDataLoader().run()
    dp1 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        # feature_selector= DummyFeatureSelector(),
        feature_selector=PCAFeatureSelector(),
        # feature_selector= DropFeatureSelector(),
        imputer=RevenueBucketImputer(),
        # imputer=DummyImputer(),
        # scope_estimator=MiniModelArmyEstimator().set_evaluator(base.DefaultRegressorEvaluator()),
        scope_estimator=GaussianProcessEstimator().set_evaluator(base.DefaultRegressorEvaluator()),
    )
    dp2 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        feature_selector=DummyFeatureSelector(),
        imputer=DummyImputer(),
        scope_estimator=MiniModelArmyEstimator(),
    )
    dp3 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=RevenueBucketImputer(),
        scope_estimator=MiniModelArmyEstimator(),
    )
    model = OxariModel()
    model.add_pipeline(scope=1, pipeline=dp1.run_pipeline(dataset, scope=1))
    model.add_pipeline(scope=2, pipeline=dp2.run_pipeline(dataset, scope=2))
    model.add_pipeline(scope=3, pipeline=dp3.run_pipeline(dataset, scope=3))
