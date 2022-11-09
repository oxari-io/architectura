from pipeline.baseline import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from scope_estimators.mini_model_army import MiniModelArmyEstimator
from preprocessors.baseline import BaselinePreprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers.baseline import DummyImputer
from feature_reducers.baseline import DummyFeatureSelector, PCAFeatureSelector, DropFeatureSelector
from scope_estimators.mma.classifier import ClassifierOptimizer
from base import OxariModel
if __name__ == "__main__":

    dp = DefaultPipeline(
        dataset=CSVDataLoader(),
        preprocessor=BaselinePreprocessor(),
        # feature_selector= DummyFeatureSelector(),
        feature_selector= PCAFeatureSelector(),
        # feature_selector= DropFeatureSelector(),
        imputer=RevenueBucketImputer(),
        # imputer=DummyImputer(),
        scope_estimator=MiniModelArmyEstimator(),
    )

    OxariModel().add_pipeline(1, dp.run_pipeline(scope=1))
    
     