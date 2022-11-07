from pipeline.baseline import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from scope_estimators.mini_model_army import MiniModelArmyEstimator
from preprocessors.baseline import BaselinePreprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers.baseline import DummyImputer
from scope_estimators.mma.classifier import ClassifierOptimizer




if __name__ == "__main__":

    dp = DefaultPipeline(dataset = CSVDataLoader(),
                        preprocessor = BaselinePreprocessor(),
                        imputer = RevenueBucketImputer(),
                        # imputer = DummyImputer(),
                        scope_estimator = MiniModelArmyEstimator())




    dp.run_pipeline()