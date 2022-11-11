from pipeline.core import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from preprocessors.core import BaselinePreprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers.core import BaselineImputer
from feature_reducers.core import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer
from scope_estimators import BaselineEstimator, GaussianProcessEstimator, MiniModelArmyEstimator
import base
from base import OxariModel
import pandas as pd

if __name__ == "__main__":

    dataset=CSVDataLoader().run()
    dp3 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=RevenueBucketImputer(),
        scope_estimator=GaussianProcessEstimator(),
    )
    dp2 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        feature_selector=DummyFeatureReducer(),
        imputer=BaselineImputer(),
        scope_estimator=MiniModelArmyEstimator(),
    )
    dp1 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        feature_selector=DummyFeatureReducer(),
        imputer=BaselineImputer(),
        scope_estimator=BaselineEstimator(),
    )
    model = OxariModel()
    model.add_pipeline(scope=1, pipeline=dp1.run_pipeline(dataset, scope=1))
    model.add_pipeline(scope=2, pipeline=dp2.run_pipeline(dataset, scope=2))
    model.add_pipeline(scope=3, pipeline=dp3.run_pipeline(dataset, scope=3))
    X = dataset.get_data_by_name("original")
    
    print("Eval results")
    print(pd.json_normalize(model.collect_eval_results()))
    print("Predict with Pipeline")
    print(dp1.predict(X))
    print("Predict with Model")
    print(model.predict(X, scope=1))
    print("Predict ALL with Model")
    print(model.predict(X))
    
    
