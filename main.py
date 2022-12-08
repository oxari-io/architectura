import time
from datetime import date
from pipeline.core import DefaultPipeline
from dataset_loader.csv_loader import CSVDataManager
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver, LocalLARModelSaver, LocalDataSaver
from preprocessors import BaselinePreprocessor, ImprovedBaselinePreprocessor, IIDPreprocessor
from postprocessors import ScopeImputerPostprocessor
from base import BaselineConfidenceEstimator, JacknifeConfidenceEstimator
from imputers import BaselineImputer, KMeansBucketImputer, RevenueBucketImputer, RevenueExponentialBucketImputer, RevenueQuantileBucketImputer, RevenueParabolaBucketImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator, LinearRegressionEstimator, BayesianRegressionEstimator
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
from pprint import pprint

if "intel" in platform.processor().lower():
    from sklearnex import patch_sklearn
    patch_sklearn()

DATA_DIR = pathlib.Path('local/data')
from lar_calculator.model_lar import OxariLARCalculator

if __name__ == "__main__":
    

    # TODO: Rename dataset
    dataset = CSVDataManager().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=BaselineEstimator(),
    )
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=BaselineEstimator(),
    )
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=BaselineEstimator(),
    )
    model = OxariMetaModel()
    scope_imputer = ScopeImputerPostprocessor(estimator=model)
    model.add_pipeline(
        scope=1,
        pipeline=dp1.optimise(SPLIT_1.train.X, SPLIT_1.train.y).fit(SPLIT_1.train.X, SPLIT_1.train.y).evaluate(SPLIT_1.rem.X, SPLIT_1.rem.y, SPLIT_1.val.X, SPLIT_1.val.y),
    )
    model.add_pipeline(
        scope=2,
        pipeline=dp2.optimise(SPLIT_2.train.X, SPLIT_2.train.y).fit(SPLIT_2.train.X, SPLIT_2.train.y).evaluate(SPLIT_2.rem.X, SPLIT_2.rem.y, SPLIT_2.val.X, SPLIT_2.val.y),
    )
    model.add_pipeline(
        scope=3,
        pipeline=dp3.optimise(SPLIT_3.train.X, SPLIT_3.train.y).fit(SPLIT_3.train.X, SPLIT_3.train.y).evaluate(SPLIT_3.rem.X, SPLIT_3.rem.y, SPLIT_3.val.X, SPLIT_3.val.y),
    )

    print("Parameter Configuration")
    pprint(dp1.get_config(deep=True))
    print(dp2.get_config(deep=True))
    print(dp3.get_config(deep=True))

    ### EVALUATION RESULTS ###
    print("Eval results")
    print(pd.json_normalize(model.collect_eval_results()))
    print("Predict with Pipeline")
    # print(dp1.predict(X))
    print("Predict with Model only SCOPE1")
    print(model.predict(X, scope=1))

    scope_imputed_data = scope_imputer.run(X=DATA)
    today = time.strftime('%d-%m-%Y')
    dataset.add_data(OxariDataManager.IMPUTED_SCOPES, scope_imputed_data, f"This data has all scopes imputed by the model on {today} at {time.localtime()}")
    print(scope_imputed_data)

    print("\n", "Predict ALL with Model")
    print(model.predict(X))

    print("\n", "Predict ALL on Mock data")
    print(model.predict(helper.mock_data()))

    print("\n", "Compute Confidences")
    confidence_intervall_estimator = JacknifeConfidenceEstimator(pipeline=dp1, n_splits=3)
    confidence_intervall_estimator = confidence_intervall_estimator.fit(SPLIT_1.train.X, SPLIT_1.train.y)
    print(confidence_intervall_estimator.predict(SPLIT_1.val.X))

    print("\n", "Predict LARs on Mock data")
    lar_model = OxariLARCalculator().fit(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    lar_imputed_data = lar_model.transform(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    dataset.add_data(OxariDataManager.IMPUTED_LARS, lar_imputed_data, f"This data has all LAR values imputed by the model on {today} at {time.localtime()}")
    print(lar_imputed_data)

    tmp_pipeline = model.get_pipeline(1)

    # tmp_pipeline.feature_selector.visualize(tmp_pipeline._preprocess(X))
    ### SAVE OBJECTS ###

    local_model_saver = LocalMetaModelSaver(today=time.strftime('%d-%m-%Y'), name="test").set(model=model)
    local_lar_saver = LocalLARModelSaver(today=time.strftime('%d-%m-%Y'), name="test").set(model=lar_model)
    local_data_saver = LocalDataSaver(today=time.strftime('%d-%m-%Y'), name="test").set(dataset=dataset)
    SavingManager = OxariSavingManager(meta_model=local_model_saver, lar_model=local_lar_saver, dataset=local_data_saver)
    SavingManager.run()
