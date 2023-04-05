import pytest
from base.common import OxariMetaModel, OxariPipeline
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.constants import DATA_DIR
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, DataFilter, Datasource, FinancialLoader, OxariDataManager, PartialLoader, ScopeLoader, SimpleDataFilter
from base.helper import LogTargetScaler, data_point, mock_data

from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
from datasources.loaders import NetZeroIndexLoader, RegionLoader
from datasources.local import LocalDatasource
from datasources.online import OnlineCSVDatasource, OnlineExcelDatasource, S3Datasource
import pandas as pd
import numpy as np
import logging
from feature_reducers.core import PCAFeatureReducer
from imputers.revenue_bucket import RevenueQuantileBucketImputer
from lar_calculator.lar_model import OxariUnboundLAR
from pipeline.core import DefaultPipeline
from postprocessors.missing_year_imputers import DerivativeMissingYearImputer
from postprocessors.scope_imputers import ScopeImputerPostprocessor
from preprocessors.core import IIDPreprocessor
from scope_estimators.mini_model_army import MiniModelArmyEstimator
from scope_estimators.svm import SupportVectorEstimator

from tests.fixtures import const_data_manager, const_pipeline, const_meta_model, const_example_df, const_example_df_multi_rows, const_example_dict, const_example_dict_multi_rows, const_example_series, const_dataset_filtered, const_data_for_scope_imputation

# logging.basicConfig(level=logging.DEBUG)
# mylogger = logging.getLogger()


def test_pipeline_t(const_data_manager: OxariDataManager):
    bag = const_data_manager.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=5),
        scope_estimator=SupportVectorEstimator(n_trials=1, n_startup_trials=1),
        ci_estmator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    assert len(dp1._evaluation_results) > 0


def test_pipeline_q(const_data_manager: OxariDataManager):
    bag = const_data_manager.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    dp1 = DefaultPipeline(
        preprocessor=IIDPreprocessor(),
        feature_reducer=PCAFeatureReducer(),
        imputer=RevenueQuantileBucketImputer(buckets_number=5),
        scope_estimator=MiniModelArmyEstimator(n_buckets=5, n_trials=2, n_startup_trials=2),
        ci_estimator=BaselineConfidenceEstimator(),
        scope_transformer=LogTargetScaler(),
    ).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)
    assert len(dp1._evaluation_results) > 0


def test_pipeline_prediction(
    const_pipeline: OxariPipeline,
    const_example_series:pd.Series,
    const_example_df:pd.DataFrame,
    const_example_df_multi_rows:pd.DataFrame,
):
    result = const_pipeline.predict(const_example_series)
    assert len(result)>0
    reference = result
    next_result = const_pipeline.predict(const_example_series)
    assert (reference == next_result).all(), f"Prediction of dict ({reference}) is not the same as for series ({next_result})"
    next_result = const_pipeline.predict(const_example_df)
    assert (reference == next_result).all(), f"Prediction of dict ({reference}) is not the same as for df ({next_result})"
    prediction_results = const_pipeline.predict(const_example_df_multi_rows)
    assert (reference==prediction_results).all()




def test_metamodel_prediction(
    const_meta_model: OxariMetaModel,
    const_example_series:pd.Series,
    const_example_dict:dict,
    const_example_dict_multi_rows:list[dict],
    const_example_df:pd.DataFrame,
    const_example_df_multi_rows:pd.DataFrame,
):
    result = const_meta_model.predict(const_example_series)
    assert len(result)>0
    reference = result.iloc[0]

    next_result = const_meta_model.predict(const_example_series).iloc[0]
    assert (reference == next_result).all(), f"Prediction of dict ({reference}) is not the same as for series ({next_result})"
    next_result = const_meta_model.predict(const_example_dict).iloc[0]
    assert (reference == next_result).all(), f"Prediction of dict ({reference}) is not the same as for dict ({next_result})"
    next_result = const_meta_model.predict(const_example_df).iloc[0]
    assert (reference == next_result).all(), f"Prediction of dict ({reference}) is not the same as for series ({next_result})"

    prediction_results = const_meta_model.predict(const_example_df_multi_rows)
    assert (reference==prediction_results).all().all()

    prediction_results = const_meta_model.predict(const_example_dict_multi_rows)
    assert (reference==prediction_results).all().all()

@pytest.mark.parametrize("data_point", [
    data_point(0.0),
    data_point(0.2),
    data_point(0.4),
    data_point(0.6),
    data_point(0.8),
    data_point(1.0),
])
def test_metamodel_prediction_with_holes(
    const_meta_model: OxariMetaModel,
    data_point:dict,
):
    result = const_meta_model.predict(data_point)
    assert len(result)>0

# TODO: Add multiple confidence tests
def test_confidences(const_meta_model:OxariMetaModel, const_data_manager: OxariDataManager):
    bag = const_data_manager.get_split_data(OxariDataManager.ORIGINAL)    
    SPLIT_1 = bag.scope_1
    results = const_meta_model.predict(SPLIT_1.val.X, return_ci=True)
    assert len(results) > 0

def test_year_interpolation(const_meta_model:OxariMetaModel, const_dataset_filtered:pd.DataFrame):
    # scope_imputer = ScopeImputerPostprocessor(estimator=const_meta_model).run(X=DATA_FOR_IMPUTE).evaluate()
    DATA = const_dataset_filtered
    # get_pipeline(1) is based on gut feeling
    data_filled = const_meta_model.get_pipeline(1).preprocessor.transform(DATA)
    data_year_imputed = DerivativeMissingYearImputer().fit_transform(data_filled)
    assert len(DATA) < len(data_year_imputed)    

def test_scope_imputation(const_meta_model:OxariMetaModel, const_dataset_filtered:pd.DataFrame):
    DATA = const_dataset_filtered
    data_filled = const_meta_model.get_pipeline(1).preprocessor.transform(DATA)
    data_year_imputed = DerivativeMissingYearImputer().fit_transform(data_filled)
    scope_imputer = ScopeImputerPostprocessor(estimator=const_meta_model).run(X=data_year_imputed).evaluate()
    assert len(scope_imputer.data) > 0  

def test_lar_imputation(const_data_for_scope_imputation:OxariDataManager):
    lar_computations = OxariUnboundLAR().fit_transform(const_data_for_scope_imputation.get_scopes(OxariDataManager.IMPUTED_SCOPES))
    assert len(lar_computations) > 0  