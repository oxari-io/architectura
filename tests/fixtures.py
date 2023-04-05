import pytest
from base.common import OxariMetaModel, OxariPipeline
from base.confidence_intervall_estimator import BaselineConfidenceEstimator

from base.dataset_loader import CompanyDataFilter, OxariDataManager
from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
import pandas as pd
from feature_reducers.core import PCAFeatureReducer
from imputers.revenue_bucket import RevenueQuantileBucketImputer

from pipeline.core import DefaultPipeline
from preprocessors.core import IIDPreprocessor
from scope_estimators.svm import SupportVectorEstimator


@pytest.fixture
def const_data_manager():
    dataset = DefaultDataManager().set_filter(CompanyDataFilter(0.1)).run()
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

def const_data_point():
    num_data = {
        "ft_numc_stock_return": -0.03294267654418,
        "ft_numc_total_assets": 0.0,
        "ft_numc_ppe": 1446082.0,
        "ft_numc_roa": 0.0145850556922605,
        "ft_numc_roe": 0.34,
        "ft_numc_total_liab": 1421.287,
        "ft_numc_equity": 1124.699,
        "ft_numc_revenue": 503.999999604178,
        "ft_numc_market_cap": 635.348579719647,
        "ft_numc_inventories": 13991.0,
        "ft_numc_net_income": 34.9999999725123,
        "ft_numc_cash": 231.043,
        "ft_numd_employees": 1000,
        "ft_numc_rd": 500,
        "ft_numc_prior_tg_numc_scope_1": 26523,
        "ft_numc_prior_tg_numc_scope_2": 50033,
        "ft_numc_prior_tg_numc_scope_3": None,
        "key_year": 2019.0,
        "key_isin": "FR0000051070",
        "tg_numc_scope_1": None,
        "tg_numc_scope_2": None,
        "tg_numc_scope_3": None,
    }

    cat_data = {
        "ft_catm_industry_name": "Industrial Conglomerates",
        "ft_catm_country_name": "Philippines",
        "ft_catm_sector_name": "Industrials",
    }
    
    return num_data,cat_data

@pytest.fixture
def const_example_series():
    num_data, cat_data = const_data_point()
    return pd.Series({**num_data, **cat_data})

@pytest.fixture
def const_example_df():
    num_data, cat_data = const_data_point()
    return pd.Series({**num_data, **cat_data}).to_frame().T.sort_index(axis=1)

@pytest.fixture
def const_example_df_multi_rows():
    num_data, cat_data = const_data_point()
    return pd.DataFrame([{**num_data, **cat_data},{**num_data, **cat_data}])

@pytest.fixture
def const_example_dict():
    num_data, cat_data = const_data_point()
    return {**num_data, **cat_data}

@pytest.fixture
def const_example_dict_multi_rows():
    num_data, cat_data = const_data_point()
    return [{**num_data, **cat_data}, {**num_data, **cat_data}]
