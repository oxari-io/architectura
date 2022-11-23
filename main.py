import time
from datetime import date
from pipeline.core import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from base import OxariDataManager, OxariSavingManager, LocalModelSaver
from preprocessors import BaselinePreprocessor
from postprocessors import ScopeImputerPostprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer, KMeansBucketImputer
from feature_reducers.core import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator
import base
from base import helper
from base import OxariModel
import pandas as pd
# import cPickle as
import joblib as pkl
import io
from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import pathlib
import platform

if "intel" in platform.processor().lower():
    from sklearnex import patch_sklearn
    patch_sklearn()    

DATA_DIR = pathlib.Path('local/data')
if __name__ == "__main__":

    dataset = CSVDataLoader().run()
    dp3 = DefaultPipeline(
        scope=3,
        preprocessor=BaselinePreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=RevenueBucketImputer(),
        scope_estimator=PredictMedianEstimator(),
    )
    dp2 = DefaultPipeline(
        scope=2,
        preprocessor=BaselinePreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=BaselineImputer(),
        scope_estimator=PredictMeanEstimator(),
    )
    dp1 = DefaultPipeline(
        scope=1,
        preprocessor=BaselinePreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=KMeansBucketImputer(),
        scope_estimator=MiniModelArmyEstimator(),
    )
    model = OxariModel()
    postprocessor = ScopeImputerPostprocessor(estimator=model)
    model.add_pipeline(scope=1, pipeline=dp1.run_pipeline(dataset))
    model.add_pipeline(scope=2, pipeline=dp2.run_pipeline(dataset))
    model.add_pipeline(scope=3, pipeline=dp3.run_pipeline(dataset))

    X = dataset.get_data_by_name("original")

    ### EVALUATION RESULTS ###
    print("Eval results")
    print(pd.json_normalize(model.collect_eval_results()))

    # print("Predict with Pipeline")
    # print(dp1.predict(X))
    # print("Predict with Model")
    # print(model.predict(X, scope=1))

    scope_inputed_data = postprocessor.run(X=X)
    today = time.strftime('%d-%m-%Y')
    dataset.add_data(OxariDataManager.IMPUTED, scope_inputed_data, f"This data has all scopes imputed by the model on {today} at {time.localtime()}")

    print("\n", "Predict ALL with Model")
    print(model.predict(X))
    
    print(model.predict(helper.mock_data()))

    ### SAVE OBJECTS ###
    
    local_model_saver = LocalModelSaver(today=time.strftime('%d-%m-%Y'), name="test").set(model=model)
    SavingManager = OxariSavingManager(meta_model=local_model_saver)
    SavingManager.run()
