import time
from datetime import date
from pipeline.core import DefaultPipeline, CVPipeline
from dataset_loader.csv_loader import CSVDataManager
from base import OxariDataManager, OxariSavingManager, LocalMetaModelSaver, LocalLARModelSaver, LocalDataSaver,S3MetaModelSaver, S3DataSaver, S3LARModelSaver
from preprocessors import BaselinePreprocessor, ImprovedBaselinePreprocessor, IIDPreprocessor, NormalizedIIDPreprocessor
from postprocessors import ScopeImputerPostprocessor, ShapExplainer, ResidualExplainer, JumpRateExplainer, DecisionExplainer
from base import BaselineConfidenceEstimator, JacknifeConfidenceEstimator
from imputers import BaselineImputer, KMeansBucketImputer, RevenueBucketImputer, RevenueExponentialBucketImputer, RevenueQuantileBucketImputer, RevenueParabolaBucketImputer
from feature_reducers import DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, IsomapFeatureSelector, MDSSelector
from scope_estimators import PredictMedianEstimator, GaussianProcessEstimator, MiniModelArmyEstimator, SupportVectorEstimator, DummyEstimator, PredictMeanEstimator, BaselineEstimator, LinearRegressionEstimator, BayesianRegressionEstimator, GLMEstimator, IndependentFeatureVotingRegressionEstimator
from base.confidence_intervall_estimator import ProbablisticConfidenceEstimator, BaselineConfidenceEstimator
import base
from base import helper
from base.helper import LogarithmScaler
from base import OxariMetaModel
import pandas as pd
# import cPickle as
import joblib as pkl
import io
from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader,S3ScopeLoader, S3CategoricalLoader,S3FinancialLoader
import pathlib
from pprint import pprint
import numpy as np
from lar_calculator.lar_model import OxariUnboundLAR
import matplotlib.pyplot as plt
DATA_DIR = pathlib.Path('local/data')
N_TRIALS = 5
N_STARTUP_TRIALS = 1
if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')

    # dataset = CSVDataManager(scope_loader=S3ScopeLoader(), financial_loader=S3FinancialLoader(), categorical_loader=S3CategoricalLoader()).run()
    dataset = CSVDataManager().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # TODO: Test what happens if not all the optimise functions are called.
    # TODO: Check why scope_transformer destroys accuracy.
    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=3),
        scope_estimator=SupportVectorEstimator(),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogarithmScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureSelector(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=SupportVectorEstimator(),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogarithmScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_1.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureSelector(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=SupportVectorEstimator(),
        ci_estimator=JacknifeConfidenceEstimator(),
        scope_transformer=LogarithmScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_1.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)

    print("Parameter Configuration")
    print(dp1.get_config(deep=True))
    print(dp2.get_config(deep=True))
    print(dp3.get_config(deep=True))

    ### EVALUATION RESULTS ###
    print("Eval results")
    eval_results = pd.json_normalize(model.collect_eval_results())
    eval_results.to_csv('local/eval_results/model_pipelines_test.csv')
    print(eval_results)

    print("Predict with Model only SCOPE1")
    print(model.predict(SPLIT_1.val.X, scope=1))

    print("Impute scopes with Model")
    scope_imputer = ScopeImputerPostprocessor(estimator=model).run(X=DATA).evaluate()
    dataset.add_data(OxariDataManager.IMPUTED_SCOPES, scope_imputer.data, f"This data has all scopes imputed by the model on {today} at {time.localtime()}")
    dataset.add_data(OxariDataManager.JUMP_RATES, scope_imputer.jump_rates, f"This data has jump rates per yearly transition of each company")
    dataset.add_data(OxariDataManager.JUMP_RATES_AGG, scope_imputer.jump_rates_agg, f"This data has summaries of jump-rates per company")
    
    scope_imputer.jump_rates.to_csv('local/eval_results/model_jump_rates_test.csv')
    scope_imputer.jump_rates_agg.to_csv('local/eval_results/model_jump_rates_agg_test.csv')

    print("\n", "Predict LARs on Mock data")
    lar_model = OxariUnboundLAR().fit(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    lar_imputed_data = lar_model.transform(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    dataset.add_data(OxariDataManager.IMPUTED_LARS, lar_imputed_data, f"This data has all LAR values imputed by the model on {today} at {time.localtime()}")
    print(lar_imputed_data)

    print("Explain Features using SHAP")
    explainer = ShapExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.val)
    explainer.visualize()
    
    print("Explain Effects of features on Residuals")
    explainer1 = ResidualExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    explainer2 = JumpRateExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    explainer3 = DecisionExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    fig, axes = explainer1.visualize()
    fig, axes = explainer2.visualize()
    fig, axes = explainer3.visualize()
    plt.show(block=True)
    
    
    print("\n", "Predict ALL with Model")
    print(model.predict(SPLIT_1.val.X))

    print("\n", "Predict ALL on Mock data")
    print(model.predict(helper.mock_data()))

    print("\n", "Compute Confidences")
    print(model.predict(SPLIT_1.val.X, return_ci=True))

    print("\n", "DIRECT COMPARISON")
    result = model.predict(SPLIT_1.test.X, scope=1, return_ci=True)
    result["true_scope"] = SPLIT_1.test.y.values
    result["absolute_difference"] = np.abs(result["pred"] - result["true_scope"])
    result["offset_ratio"] = np.maximum(result["pred"], result["true_scope"]) / np.minimum(result["pred"], result["true_scope"]) 
    result.loc[:, SPLIT_1.test.X.columns] = SPLIT_1.test.X.values 
    result.to_csv('local/eval_results/model_training_test.csv')
    print(result)



    tmp_pipeline = model.get_pipeline(1)

    # tmp_pipeline.feature_selector.visualize(tmp_pipeline._preprocess(X))
    ### SAVE OBJECTS ###

    local_model_saver = LocalMetaModelSaver(time=time.strftime('%d-%m-%Y'), name="test").set(model=model)
    local_lar_saver = LocalLARModelSaver(time=time.strftime('%d-%m-%Y'), name="test").set(model=lar_model)
    local_data_saver = LocalDataSaver(time=time.strftime('%d-%m-%Y'), name="test").set(dataset=dataset)
    # local_model_saver = S3MetaModelSaver(time=time.strftime('%d-%m-%Y'), name="test").set(model=model)
    # local_lar_saver = S3LARModelSaver(time=time.strftime('%d-%m-%Y'), name="test").set(model=lar_model)
    # local_data_saver = S3DataSaver(time=time.strftime('%d-%m-%Y'), name="test").set(dataset=dataset)
    SavingManager = OxariSavingManager(meta_model=local_model_saver, lar_model=local_lar_saver, dataset=local_data_saver)
    SavingManager.run()
