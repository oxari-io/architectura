import time
from matplotlib import pyplot as plt

from sklearn.preprocessing import PowerTransformer
from base.common import OxariMetaModel
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.constants import FEATURE_SET_VIF_UNDER_05, FEATURE_SET_VIF_UNDER_10, FEATURE_SET_VIF_UNDER_15, FEATURE_SET_VIF_UNDER_20, FEATURE_SET_VIF_UNDER_25
from base.dataset_loader import OxariDataManager
from base.helper import ArcSinhScaler, LogTargetScaler
from base.run_utils import get_small_datamanager_configuration

import pathlib
import pickle
from datastores.saver import LocalDestination, OxariSavingManager, PickleSaver
from feature_reducers.core import DummyFeatureReducer, SelectionFeatureReducer
from imputers.core import DummyImputer
from imputers.revenue_bucket import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from postprocessors.core import ALEExplainer, DecisionExplainer, JumpRateExplainer, PDExplainer, PDVarianceExplainer, PermutationImportanceExplainer, ResidualExplainer, ShapExplainer
from preprocessors.core import IIDPreprocessor, NormalizedIIDPreprocessor
from scope_estimators.k_neighbors import KNNEstimator
from scope_estimators.linear_models import LinearRegressionEstimator
from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator, MiniModelArmyEstimator

DATA_DIR = pathlib.Path('model-data/data/input')

DATE_FORMAT = 'T%Y%m%d'

N_TRIALS = 40
N_STARTUP_TRIALS = 20
STAGE = "p_"


def train_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset, feature_set):
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=NormalizedIIDPreprocessor(fin_transformer=ArcSinhScaler()),
        feature_reducer=SelectionFeatureReducer(feature_set),
        imputer=DummyImputer(),
        scope_estimator=EvenWeightMiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    # dp2 = DefaultPipeline(
    #     preprocessor=NormalizedIIDPreprocessor(fin_transformer=ArcSinhScaler()),
    #     feature_reducer=SelectionFeatureReducer(feature_set),
    #     imputer=DummyImputer(),
    #     scope_estimator=EvenWeightMiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
    #     ci_estimator=BaselineConfidenceEstimator(),
    #     scope_transformer=LogTargetScaler(),
    # ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    # dp3 = DefaultPipeline(
    #     preprocessor=NormalizedIIDPreprocessor(fin_transformer=ArcSinhScaler()),
    #     feature_reducer=SelectionFeatureReducer(feature_set),
    #     imputer=DummyImputer(),
    #     scope_estimator=EvenWeightMiniModelArmyEstimator(10, n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
    #     ci_estimator=BaselineConfidenceEstimator(),
    #     scope_transformer=LogTargetScaler(),
    # ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    # model.add_pipeline(scope=2, pipeline=dp2)
    # model.add_pipeline(scope=3, pipeline=dp3)
    return model

def train_simple_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset, feature_set):
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    # Test what happens if not all the optimise functions are called.
    dp1 = DefaultPipeline(
        preprocessor=NormalizedIIDPreprocessor(fin_transformer=ArcSinhScaler()),
        feature_reducer=SelectionFeatureReducer(feature_set),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=KNNEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    # dp2 = DefaultPipeline(
    #     preprocessor=NormalizedIIDPreprocessor(fin_transformer=ArcSinhScaler()),
    #     feature_reducer=SelectionFeatureReducer(feature_set),
    #     imputer=RevenueQuantileBucketImputer(10),
    #     scope_estimator=KNNEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
    #     ci_estimator=BaselineConfidenceEstimator(),
    #     scope_transformer=LogTargetScaler(),
    # ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    # dp3 = DefaultPipeline(
    #     preprocessor=NormalizedIIDPreprocessor(fin_transformer=ArcSinhScaler()),
    #     feature_reducer=SelectionFeatureReducer(feature_set),
    #     imputer=RevenueQuantileBucketImputer(10),
    #     scope_estimator=KNNEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
    #     ci_estimator=BaselineConfidenceEstimator(),
    #     scope_transformer=LogTargetScaler(),
    # ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    # model.add_pipeline(scope=2, pipeline=dp2)
    # model.add_pipeline(scope=3, pipeline=dp3)
    return model


if __name__ == "__main__":
    # DATE = time.strftime(DATE_FORMAT)
    DATE = "T20240808"

    dataset = get_small_datamanager_configuration(1).run()
    FEATURE_SET_ALL = dataset.get_split_data(OxariDataManager.ORIGINAL).scope_1.train[0].columns.tolist()

    feature_sets = {
        # "ft_set_vif_05":FEATURE_SET_VIF_UNDER_05, 
        # "ft_set_vif_10":FEATURE_SET_VIF_UNDER_10, 
        # "ft_set_vif_15":FEATURE_SET_VIF_UNDER_15,
        "ft_set_vif_all":FEATURE_SET_ALL,
        "ft_set_vif_20":FEATURE_SET_VIF_UNDER_20,
        "ft_set_vif_25":FEATURE_SET_VIF_UNDER_25,
    }
    for fs_name, feature_set in feature_sets.items():
        model = train_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset, feature_set) 
        PickleSaver().set_time(DATE).set_extension(".pkl").set_name(f"p_model_fi_{fs_name}").set_object(model).set_datatarget(LocalDestination(path="model-data/output")).save()



    for fs_name, feature_set in feature_sets.items():
        cwd = pathlib.Path(__file__).parent
        model = pickle.load((cwd.parent / f'model-data/output/{DATE}-p_model_fi_{fs_name}.pkl').open('rb'))
        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3
        explainer0 = ShapExplainer(model.get_pipeline(1), sample_size=1000).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
        shap_package = (explainer0.shap_values, explainer0.X, explainer0.y)
        PickleSaver().set_time(DATE).set_extension(".pkl").set_name(f"p_model_fi_shap_{fs_name}").set_object(shap_package).set_datatarget(LocalDestination(path="model-data/output")).save()
        
   