import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from base import (LocalDataSaver, LocalLARModelSaver, LocalMetaModelSaver,
                  OxariDataManager, OxariMetaModel, OxariSavingManager, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.helper import LogarithmScaler
from datasources.core import PreviousScopeFeaturesDataManager
from feature_reducers import DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from postprocessors import (DecisionExplainer, JumpRateExplainer,
                            ResidualExplainer, ScopeImputerPostprocessor,
                            ShapExplainer)
from preprocessors import BaselinePreprocessor
from scope_estimators import MiniModelArmyEstimator

DATA_DIR = pathlib.Path('local/data')
from lar_calculator.lar_model import OxariUnboundLAR

N_TRIALS = 40
N_STARTUP_TRIALS = 10

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')

    dataset = PreviousScopeFeaturesDataManager().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=3),
        scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogarithmScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=3),
        scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogarithmScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_1.train)
    dp3 = DefaultPipeline(
        preprocessor=BaselinePreprocessor(),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=3),
        scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
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
    print(eval_results)
    eval_results.to_csv('local/eval_results/model_pipelines.csv')
    print("Predict with Pipeline")
    # print(dp1.predict(X))
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

    print("Explain Effects of features")
    explainer0 = ShapExplainer(model.get_pipeline(1), sample_size=100).fit(*SPLIT_1.train).explain(*SPLIT_1.val)
    fig, ax = explainer0.visualize()
    fig.savefig(f'local/eval_results/importance_explainer{0}.png')
    explainer1 = ResidualExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    explainer2 = JumpRateExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    explainer3 = DecisionExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    for idx, expl in enumerate([explainer1, explainer2, explainer3]):
        fig, ax = expl.plot_tree()
        fig.savefig(f'local/eval_results/tree_explainer{idx+1}.png', dpi=600)
        fig, ax = expl.plot_importances()
        fig.savefig(f'local/eval_results/importance_explainer{idx+1}.png')

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
    result.to_csv('local/eval_results/model_training_direct_comparison.csv')
    print(result)

    print("\n", "Predict LARs on Mock data")
    lar_model = OxariUnboundLAR().fit(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    lar_imputed_data = lar_model.transform(dataset.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    dataset.add_data(OxariDataManager.IMPUTED_LARS, lar_imputed_data, f"This data has all LAR values imputed by the model on {today} at {time.localtime()}")
    print(lar_imputed_data)

    tmp_pipeline = model.get_pipeline(1)

    # tmp_pipeline.feature_selector.visualize(tmp_pipeline._preprocess(X))
    ### SAVE OBJECTS ###

    local_model_saver = LocalMetaModelSaver(time=time.strftime('%d-%m-%Y'), name="lightweight").set(model=model)
    local_lar_saver = LocalLARModelSaver(time=time.strftime('%d-%m-%Y'), name="lightweight").set(model=lar_model)
    local_data_saver = LocalDataSaver(time=time.strftime('%d-%m-%Y'), name="lightweight").set(dataset=dataset)
    SavingManager = OxariSavingManager(meta_model=local_model_saver, lar_model=local_lar_saver, dataset=local_data_saver)
    SavingManager.run()
