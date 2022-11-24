from pipeline.core import DefaultPipeline
from dataset_loader.csv_loader import CSVDataLoader
from base import OxariDataManager
from preprocessors import BaselinePreprocessor
from postprocessors import ScopeImputerPostprocessor
from imputers.revenue_bucket import RevenueBucketImputer
from imputers.core import BaselineImputer
from feature_reducers.core import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, DummyEstimator, PredictMeanEstimator
import base
from base import OxariModel
import pandas as pd
# import cPickle as
import joblib as pkl
import io

if __name__ == "__main__":

    dataset = CSVDataLoader().run()
    dp3 = DefaultPipeline(
        scope=3,
        preprocessor=BaselinePreprocessor(),
        feature_selector=PCAFeatureSelector(),
        imputer=BaselineImputer(),
        scope_estimator=DummyEstimator(),
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
        scope_estimator=MiniModelArmyEstimator(),
    )
    model = OxariModel()
    postprocessor = ScopeImputerPostprocessor(model)
    model.add_pipeline(scope=1, pipeline=dp1.run_pipeline(dataset))
    # model.add_pipeline(scope=2, pipeline=dp2.run_pipeline(dataset))
    # model.add_pipeline(scope=3, pipeline=dp3.run_pipeline(dataset))
    X = dataset.get_data_by_name("original")
    X = X.drop(columns=["isin"])

    print("Eval results")
    print(pd.json_normalize(model.collect_eval_results()))
    print("Predict with Pipeline")
    # print(dp1.predict(X))
    # print("Predict with Model")
    # print(model.predict(X, scope=1))

    scope_inputed_data = postprocessor.run(X=X)
    #  TODO: Add timestamp to the description of imputed scopes.
    dataset.add_data(OxariDataManager.IMPUTED, scope_inputed_data,
                     "This data has all scopes imputed by the model.")
    print("Predict ALL with Model")
    print(model.predict(X))
    pkl.dump(model, io.open('model.pkl', 'wb'))
    pkl.dump(dataset, io.open('dataset.pkl', 'wb'))
