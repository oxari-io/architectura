import time
from matplotlib import pyplot as plt

from sklearn.preprocessing import PowerTransformer
from base.common import OxariMetaModel
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.dataset_loader import OxariDataManager
from base.helper import LogTargetScaler
from base.run_utils import get_small_datamanager_configuration

import pathlib
import pickle
from datastores.saver import LocalDestination, OxariSavingManager, PickleSaver
from feature_reducers.core import DummyFeatureReducer
from imputers.revenue_bucket import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from postprocessors.core import ALEExplainer, DecisionExplainer, JumpRateExplainer, PDExplainer, PDVarianceExplainer, PermutationImportanceExplainer, ResidualExplainer, ShapExplainer
from preprocessors.core import IIDPreprocessor
from scope_estimators.k_neighbors import KNNEstimator
from scope_estimators.linear_models import LinearRegressionEstimator
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

def train_simple_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset):
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
        scope_estimator=KNNEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=KNNEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
        feature_reducer=DummyFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(10),
        scope_estimator=KNNEstimator(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)
    return model


if __name__ == "__main__":

    # cwd = pathlib.Path(__file__).parent
    # model = pickle.load((cwd.parent / 'model-data/output/T20231113_p_model_experiment_feature_impact.pkl').open('rb'))

    dataset = get_small_datamanager_configuration(1).run()

    model = train_model_for_imputation(N_TRIALS, N_STARTUP_TRIALS, dataset) 

    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    explainer2 = ALEExplainer(model.get_pipeline(1), target_name="tg_numc_scope_1", sample_size=5000).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    package_ale = (explainer2.ale_importance, explainer2.X, explainer2.y)
    PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model_experiment_feature_impact_explainer_ale").set_object(package_ale).set_datatarget(LocalDestination(path="model-data/output")).save()

    explainer3 = PDExplainer(model.get_pipeline(1), target_name="tg_numc_scope_1", sample_size=5000).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    package_pd = (explainer3.pd_importance, explainer3.X, explainer3.y)
    PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model_experiment_feature_impact_explainer_pd").set_object(package_pd).set_datatarget(LocalDestination(path="model-data/output")).save()
    
    explainer4 = PermutationImportanceExplainer(model.get_pipeline(1), target_name="tg_numc_scope_1", sample_size=5000).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    package_permut = (explainer4.permut_importance, explainer4.X, explainer4.y)
    PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model_experiment_feature_impact_explainer_permut").set_object(package_permut).set_datatarget(LocalDestination(path="model-data/output")).save()

    explainer0 = ShapExplainer(model.get_pipeline(1), sample_size=5000).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    shap_package = (explainer0.shap_values, explainer0.X, explainer0.y)
    PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model_experiment_feature_impact_explainer_shap").set_object(shap_package).set_datatarget(LocalDestination(path="model-data/output")).save()

    explainer1 = PDVarianceExplainer(model.get_pipeline(1), target_name="tg_numc_scope_1", sample_size=5000).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    package_pdv = (explainer1.pdv_importance, explainer1.X, explainer1.y)
    PickleSaver().set_time(time.strftime(DATE_FORMAT)).set_extension(".pkl").set_name("p_model_experiment_feature_impact_explainer_pdv").set_object(package_pdv).set_datatarget(LocalDestination(path="model-data/output")).save()


    # all_meta_models = [
    # ]

    # SavingManager = OxariSavingManager(*all_meta_models, )
    # SavingManager.run() 

    # fig.savefig(f'local/eval_results/importance_explainer{0}.png')
    # explainer1 = ResidualExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # explainer2 = JumpRateExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)
    # explainer3 = DecisionExplainer(model.get_pipeline(1), sample_size=10).fit(*SPLIT_1.train).explain(*SPLIT_1.test)

    # for intervall_group, expl in enumerate([explainer1, explainer2, explainer3]):
    #     fig, ax = expl.plot_tree()
    #     fig.savefig(f'local/eval_results/tree_explainer{intervall_group+1}.png', dpi=600)
    #     fig, ax = expl.plot_importances()
    #     fig.savefig(f'local/eval_results/importance_explainer{intervall_group+1}.png')