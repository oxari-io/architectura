import pytest
from base.common import OxariMetaModel, OxariPipeline
from base.confidence_intervall_estimator import BaselineConfidenceEstimator

from base.dataset_loader import CompanyDataFilter, OxariDataManager
from base.helper import LogTargetScaler, data_point
from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
import pandas as pd
from feature_reducers.core import PCAFeatureReducer
from imputers.revenue_bucket import RevenueQuantileBucketImputer

from pipeline.core import DefaultPipeline
from postprocessors.missing_year_imputers import DerivativeMissingYearImputer
from postprocessors.scope_imputers import ScopeImputerPostprocessor
from preprocessors.core import IIDPreprocessor
from scope_estimators.svm import SupportVectorEstimator


@pytest.fixture
def const_data_manager():
    dataset = DefaultDataManager().set_filter(CompanyDataFilter(0.05)).run()
    return dataset


@pytest.fixture
def const_dataset_filtered(const_data_manager: OxariDataManager):
    DATA = const_data_manager.get_data_by_name(OxariDataManager.ORIGINAL)
    return DATA


@pytest.fixture
def const_dataset_full():
    DATA = DefaultDataManager().run().get_data_by_name(OxariDataManager.ORIGINAL)
    return DATA


@pytest.fixture
def const_pipeline(const_data_manager: OxariDataManager):
    bag = const_data_manager.get_split_data(OxariDataManager.ORIGINAL)

    SPLIT_1 = bag.scope_1

    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=5),
        scope_estimator=SupportVectorEstimator(n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)

    return dp1


@pytest.fixture
def const_meta_model(const_data_manager: OxariDataManager, const_pipeline: OxariPipeline):
    bag = const_data_manager.get_split_data(OxariDataManager.ORIGINAL)

    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    dp1 = const_pipeline
    dp2 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=SupportVectorEstimator(n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_2.train).fit(*SPLIT_2.train).evaluate(*SPLIT_2.rem, *SPLIT_2.val).fit_confidence(*SPLIT_2.train)
    dp3 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(),
        scope_estimator=SupportVectorEstimator(n_trials=1, n_startup_trials=1),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_3.train).fit(*SPLIT_3.train).evaluate(*SPLIT_3.rem, *SPLIT_3.val).fit_confidence(*SPLIT_3.train)

    model = OxariMetaModel()
    model.add_pipeline(scope=1, pipeline=dp1)
    model.add_pipeline(scope=2, pipeline=dp2)
    model.add_pipeline(scope=3, pipeline=dp3)
    return model


@pytest.fixture
def const_example_series():
    return pd.Series(data_point())

@pytest.fixture
def const_example_df():
    return pd.Series(data_point()).to_frame().T.sort_index(axis=1)

@pytest.fixture
def const_example_df_multi_rows():
    d_point = data_point()
    return pd.DataFrame([d_point,d_point])

@pytest.fixture
def const_example_dict():
    return data_point()

@pytest.fixture
def const_example_dict_multi_rows():
    d_point = data_point()
    return [d_point, d_point]


@pytest.fixture
def const_data_for_scope_imputation(const_meta_model:OxariMetaModel, const_data_manager:OxariDataManager):
    DATA = const_data_manager.get_data_by_name(OxariDataManager.ORIGINAL)
    data_filled = const_meta_model.get_pipeline(1).preprocessor.transform(DATA)
    data_year_imputed = DerivativeMissingYearImputer().fit_transform(data_filled)
    scope_imputer = ScopeImputerPostprocessor(estimator=const_meta_model).run(X=data_year_imputed).evaluate()
    const_data_manager.add_data(OxariDataManager.IMPUTED_SCOPES, scope_imputer.data, f"This data has all scopes imputed by the model")
    return const_data_manager