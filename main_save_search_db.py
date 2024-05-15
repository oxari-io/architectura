import io
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from base import (OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, EmptyLoader, FinancialLoader, ScopeLoader
from base.helper import LogTargetScaler
from base.run_utils import compute_jump_rates, compute_lar, get_deduplicated_datamanager_configuration, impute_missing_years, impute_scopes
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from datasources.loaders import NetZeroIndexLoader, RegionLoader
from datastores.saver import CSVSaver, LocalDestination, MongoDestination, MongoSaver, OxariSavingManager, PickleSaver, S3Destination
from feature_reducers import DummyFeatureReducer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from postprocessors import (DecisionExplainer, JumpRateExplainer, ResidualExplainer, ScopeImputerPostprocessor, ShapExplainer)
from preprocessors import BaselinePreprocessor, IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator
from datasources.online import S3Datasource
from datasources.local import LocalDatasource
from pymongo import ASCENDING, TEXT
import pickle as pkl

from lar_calculator.lar_model import OxariUnboundLAR

MODEL_OUTPUT_DIR = pathlib.Path('model-data/output')

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')
    model = pkl.load(io.open(MODEL_OUTPUT_DIR / 'T20240508_p_model_scope_imputation.pkl', 'rb'))
    dataset = get_deduplicated_datamanager_configuration().set_filter(CompanyDataFilter(1)).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)

    data_to_impute = DATA.copy()
    data_to_impute = impute_missing_years(data_to_impute)
    scope_imputer, imputed_data = impute_scopes(model, data_to_impute)
    lar_model, lar_imputed_data = compute_lar(imputed_data)
    jump_rate_evaluator, jump_rates = compute_jump_rates(imputed_data)

    keys = {
        "key_ticker": "text",
        "meta_name": "text",
        "meta_name_original": "text",
        "meta_country": "text",
        "meta_country_code": "text",
        "meta_symbol": "text",
        "meta_mic_code": "text",
        "meta_exchange_code_unknown": "text",
        "meta_description": "text",
        "meta_industry_name": "text",
        "meta_sector_name": "text",
        "ft_catm_region": "text",
        "ft_catm_sub_region": "text"
    }
    # Bulk text -> 1 | Enable filter with search -> 5 | Company rough identifier -> 10 | Exact Identfier -> 15 
    options = {
        "weights": {
            "key_ticker": 15,
            "meta_name": 10,
            "meta_name_before_cleaning": 10,
            "meta_name_after_normalization": 10,
            "meta_name_before_cross_merge": 10,
            "meta_country": 5,
            "meta_country_code": 5,
            "meta_symbol": 10,
            "meta_mic_code": 5,
            "meta_exchange_code_unknown": 10,
            "meta_description": 1,
            "meta_industry_name": 5,
            "meta_sector_name": 5,
            "ft_catm_region": 5,
            "ft_catm_sub_region": 5,
        },
        "name": "TextIndex"
    }

    # Data prepared for saving
    ld_fin = FinancialLoader(datasource=LocalDatasource(path="model-data/input/financials.csv")).load()
    ld_scp = ScopeLoader(datasource=LocalDatasource(path="model-data/input/scopes.csv")).load()
    ld_cat = CategoricalLoader(datasource=LocalDatasource(path="model-data/input/categoricals.csv")).load()
    ld_reg = RegionLoader().load()
    cmb_ld = EmptyLoader()

    cmb_ld = ld_fin + ld_scp + ld_cat + ld_reg

    df = DATA.filter(regex="^(ft_cat|meta_|key_ticker|key_country_code)", axis=1)
    ticker_lists = df.groupby("meta_name")["meta_other_ticker_list"].apply(sum)
    df = df.drop(columns=["meta_other_ticker_list"]).drop_duplicates(subset=["meta_name"]).merge(ticker_lists, on="meta_name")
    columns_order = ["key_ticker", "meta_name", 'ft_catm_industry_name', 'ft_catm_sector_name', 'ft_catm_country_code', 'ft_catm_region', 'ft_catm_sub_region']
    df = df[columns_order+list(df.columns.difference(columns_order))].drop(columns=['meta_exchange_x', 'meta_name_x', 'meta_symbol_x']).merge(ld_cat._data, on="key_ticker", suffixes=(None, "_DROP"))
    df = df[[c for c in df.columns if not c.endswith("_DROP")]]
    # df[df.select_dtypes('object').columns] = df[df.select_dtypes('object').columns].fillna("NA")
    # df[df.select_dtypes('category').columns] = df[df.select_dtypes('category').columns].astype(str).replace("nan", "NA")

    df_fin = ld_fin.data

    df_scope_stats = DATA.groupby(['key_year', 'ft_catm_industry_name', 'ft_catm_sector_name', 'ft_catm_country_code', 'ft_catm_region', 'ft_catm_sub_region']).median().reset_index() #.fillna("NA")

    dateformat = 'T%Y%m%d'
    all_data_features = [
        # CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_companies").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        # CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_financials").set_object(df_fin).set_datatarget(LocalDestination(path="model-data/output")),
        # CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_stats").set_object(df_scope_stats).set_datatarget(
        #     LocalDestination(path="model-data/output")),
        # CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_companies").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        # CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_financials").set_object(df_fin).set_datatarget(S3Destination(path="model-data/output")),
        # CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_stats").set_object(df_scope_stats).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime(dateformat)).set_name("p_companies").set_object(df).set_datatarget(
            MongoDestination(index=keys, path="model-data/output", options=options)),
        MongoSaver().set_time(time.strftime(dateformat)).set_name("p_financials").set_object(df_fin).set_datatarget(
            MongoDestination(path="model-data/output", index={
                "key_ticker": ASCENDING,
                "key_year": ASCENDING
            })),
        MongoSaver().set_time(time.strftime(dateformat)).set_name("p_scope_stats").set_object(df_scope_stats).set_datatarget(
            MongoDestination(path="model-data/output",
                             index={
                                 "key_year": ASCENDING,
                                 "ft_catm_industry": ASCENDING,
                                 "ft_catm_region": ASCENDING,
                                 "ft_catm_country_code": ASCENDING,
                             })),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_imputations").set_object(imputed_data).set_datatarget(
            LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_imputations").set_object(imputed_data).set_datatarget(
            S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime(dateformat)).set_name("p_scope_imputations").set_object(imputed_data).set_datatarget(
            MongoDestination(path="model-data/output", index={
                "key_ticker": ASCENDING,
                "key_year": ASCENDING
            })),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_lar_imputations").set_object(lar_imputed_data).set_datatarget(
            LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_lar_imputations").set_object(lar_imputed_data).set_datatarget(
            S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime(dateformat)).set_name("p_lar_imputations").set_object(lar_imputed_data).set_datatarget(
            MongoDestination(path="model-data/output", index={"key_ticker": ASCENDING})),
        PickleSaver().set_time(time.strftime(dateformat)).set_extension(".pkl").set_name("p_lar").set_object(lar_model).set_datatarget(LocalDestination(path="model-data/output")),
        PickleSaver().set_time(time.strftime(dateformat)).set_extension(".pkl").set_name("p_lar").set_object(lar_model).set_datatarget(S3Destination(path="model-data/output")),
    ]

    SavingManager = OxariSavingManager(*all_data_features, )
    SavingManager.run()