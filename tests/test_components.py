from typing import Type
# from postprocessors.core import DecisionExplainer, JumpRateExplainer, ResidualExplainer, TreeBasedExplainerMixin
import pytest
from base.common import OxariPreprocessor, OxariFeatureReducer, OxariImputer
from base.helper import LogTargetScaler
from feature_reducers.core import AgglomerateFeatureReducer, DummyFeatureReducer, PCAFeatureReducer
from imputers.categorical import CategoricalStatisticsImputer, HybridCategoricalStatisticsImputer
from imputers.core import BaselineImputer, DummyImputer
from imputers.iterative import MVEImputer
from imputers.kcluster_bucket import KNNBucketImputer, KMedianBucketImputer
from imputers.revenue_bucket import RevenueBucketImputer, RevenueQuantileBucketImputer
from preprocessors.core import BaselinePreprocessor, DummyPreprocessor, FastIndustryNormalisationBaselinePreprocessor, IIDPreprocessor, ImprovedBaselinePreprocessor, NormalizedIIDPreprocessor
from base.dataset_loader import OxariDataManager
from tests.fixtures import const_data_manager, const_pipeline, const_meta_model, const_example_df, const_example_df_multi_rows, const_example_dict, const_example_dict_multi_rows, const_example_series, const_dataset_filtered, const_data_for_scope_imputation
import numpy as np
import pandas as pd


@pytest.mark.parametrize("preprocessor", [IIDPreprocessor(), BaselinePreprocessor(), ImprovedBaselinePreprocessor(), NormalizedIIDPreprocessor(), FastIndustryNormalisationBaselinePreprocessor()])
def test_preprocessors(preprocessor: OxariPreprocessor, const_data_manager: OxariDataManager):
    bag = const_data_manager.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    result = preprocessor.set_imputer(DummyImputer()).fit_transform(*SPLIT_1.train)
    assert len(result) > 0


@pytest.mark.parametrize("feature_reducer", [PCAFeatureReducer(), AgglomerateFeatureReducer()])
def test_feature_reducers(feature_reducer: OxariFeatureReducer, const_data_manager: OxariDataManager):
    bag = const_data_manager.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    data_prep = DummyPreprocessor().set_imputer(DummyImputer()).fit_transform(*SPLIT_1.train)
    data_reduced = feature_reducer.fit_transform(data_prep)
    assert len(data_reduced) > 0
    assert data_reduced.shape[1] < data_prep.shape[1]


@pytest.mark.parametrize("imputer", [
    RevenueBucketImputer(),
    RevenueQuantileBucketImputer(),
    CategoricalStatisticsImputer(),
    KNNBucketImputer(),
    DummyImputer(),
    BaselineImputer(),
    HybridCategoricalStatisticsImputer(),
])
def test_imputers(imputer: OxariImputer, const_dataset_filtered: pd.DataFrame):
    data_prep: pd.DataFrame = imputer.fit_transform(const_dataset_filtered)
    assert len(data_prep) > 0
    assert data_prep.filter(regex="ft_num", axis=1).isna().sum().sum() == 0


