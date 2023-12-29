import pathlib
import time
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from base.common import OxariMetaModel
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.dataset_loader import OxariDataManager
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration
from datastores.saver import LocalDestination, OxariSavingManager, PickleSaver
from feature_reducers.core import DummyFeatureReducer
from imputers.revenue_bucket import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors.core import IIDPreprocessor
from scope_estimators.mini_model_army import MiniModelArmyEstimator

DATA_DIR = pathlib.Path('model-data/data/input')

DATE_FORMAT = 'T%Y%m%d'

N_TRIALS = 40
N_STARTUP_TRIALS = 20
STAGE = "p_"

def train_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset):
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=MiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)
    return model

if __name__ == "__main__":
    today = time.strftime(DATE_FORMAT)
    now = time.strftime('T%Y%m%d%H%M')

    dataset = get_default_datamanager_configuration().run()
    # Scope Imputation model 
    model_si = train_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset) 

    print("Eval results")
    eval_results_1 = pd.json_normalize(model_si.collect_eval_results())
    eval_results_1.T.to_csv(f'local/prod_runs/model_pipelines_{now}.csv')

    all_meta_models = [
        PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model_experiment_feature_impact").set_object(model_si).set_datatarget(LocalDestination(path="model-data/output"))
    ]

    SavingManager = OxariSavingManager(*all_meta_models, )
    SavingManager.run() 