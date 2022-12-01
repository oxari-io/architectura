import time
from datetime import date
from pipeline.core import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver
from preprocessors import BaselinePreprocessor
from postprocessors import ScopeImputerPostprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers import BaselineImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator
import base
from base import helper
from base import OxariMetaModel
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
from lar_calculator.model_lar import OxariLARCalculator
if __name__ == "__main__":

    dataset = CSVDataLoader().run()
    dp3 = DefaultPipeline(
        scope=3,
        preprocessor=BaselinePreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=BaselineImputer(),
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
        feature_selector=MDSSelector(),
        imputer=BaselineImputer(),
        scope_estimator=BaselineEstimator(),
    )
    model = OxariMetaModel()
    postprocessor = ScopeImputerPostprocessor(estimator=model)
    model.add_pipeline(scope=1, pipeline=dp1.run_pipeline(dataset))
    model.add_pipeline(scope=2, pipeline=dp2.run_pipeline(dataset))
    model.add_pipeline(scope=3, pipeline=dp3.run_pipeline(dataset))


    X = dataset.get_data_by_name(OxariDataManager.ORIGINAL)

    ### EVALUATION RESULTS ###
    print("Eval results")
    print(pd.json_normalize(model.collect_eval_results()))
    print("Predict with Pipeline")
    # print(dp1.predict(X))
    print("Predict with Model only SCOPE1")
    print(model.predict(X, scope=1))

    scope_imputed_data = postprocessor.run(X=X)
    today = time.strftime('%d-%m-%Y')
    dataset.add_data(OxariDataManager.IMPUTED_SCOPES, scope_imputed_data, f"This data has all scopes imputed by the model on {today} at {time.localtime()}")

    

    print("\n", "Predict ALL with Model")
    print(model.predict(X))

    print("\n", "Predict ALL on Mock data")
    print(model.predict(helper.mock_data()))


    print("\n", "Predict LARs on Mock data")
    lar_model = OxariLARCalculator().fit(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    lar_imputed_data = lar_model.transform(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))

    print(lar_imputed_data)

    model.get_pipeline(1).feature_selector.visualize(X)
    ### SAVE OBJECTS ###
    
    local_model_saver = LocalMetaModelSaver(today=time.strftime('%d-%m-%Y'), name="test").set(model=model)
    SavingManager = OxariSavingManager(meta_model=local_model_saver)
    SavingManager.run()
