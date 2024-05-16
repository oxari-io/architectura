import io
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from base import (OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.dataset_loader import CompanyDataFilter, EmptyLoader, FinancialLoader, ScopeLoader
from base.helper import LogTargetScaler
from base.run_utils import compute_jump_rates, compute_lar, get_deduplicated_datamanager_configuration, impute_missing_years, impute_scopes
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from datasources.loaders import ExchangePrioritizedMetaLoader, NetZeroIndexLoader, RegionLoader
from datasources.online import CachingS3Datasource
from datastores.saver import CSVSaver, CompressedCSVSaver, LocalDestination, MongoDestination, MongoSaver, OxariSavingManager, PickleSaver, S3Destination
from datasources.local import LocalDatasource
from pymongo import ASCENDING, TEXT
import pickle as pkl

from lar_calculator.lar_model import OxariUnboundLAR

MODEL_OUTPUT_DIR = pathlib.Path('model-data/output')

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')

    # Data prepared for saving
    ld_fin = FinancialLoader(datasource=CachingS3Datasource(path="model-data/input/financials.csv")).load()
    ld_scp = ScopeLoader(datasource=CachingS3Datasource(path="model-data/input/scopes.csv")).load()
    ld_cat = ExchangePrioritizedMetaLoader(datasource=CachingS3Datasource(path="model-data/input/categoricals.csv")).load()
    ld_reg = RegionLoader().load()
    cmb_ld = EmptyLoader()

    cmb_ld = ld_fin + ld_scp + ld_cat + ld_reg

    # Get Input Data
    dataset = get_default_datamanager_configuration().set_filter(CompanyDataFilter(0.01)).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    model = pkl.load(io.open(MODEL_OUTPUT_DIR / 'T20240508_q_model_scope_imputation.pkl', 'rb'))

    data_to_impute = DATA.copy()
    data_to_impute = impute_missing_years(data_to_impute)
    scope_imputer, df_imputed_data = impute_scopes(model, data_to_impute)
    lar_model, lar_imputed_data = compute_lar(df_imputed_data)
    jump_rate_evaluator, jump_rates = compute_jump_rates(df_imputed_data)

    keys = {
        "key_ticker": "text",
        "meta_name": "text",
        "meta_name_before_normalization": "text",
        "meta_name_before_extrapolation": "text",
        "meta_country": "text",
        "meta_country_code": "text",
        "meta_symbol": "text",
        "meta_mic_code": "text",
        "meta_exchange_code_unknown": "text",
        "meta_exchange_display_name": "text",
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
            "meta_name_before_normalization": 10,
            "meta_name_before_extrapolation": 10,
            "meta_country": 5,
            "meta_country_code": 5,
            "meta_symbol": 10,
            "meta_mic_code": 5,
            "meta_exchange_code_unknown": 10,
            "meta_exchange_display_name": 10,
            "meta_description": 1,
            "meta_industry_name": 5,
            "meta_sector_name": 5,
            "ft_catm_region": 5,
            "ft_catm_sub_region": 5,
        },
        "name": "TextIndex"
    }

    df = ld_cat.data[ld_cat.data["key_ticker"].isin(ld_fin.data["key_ticker"].unique().tolist())]
    df_fin = ld_fin.data
    df_scope_stats = DATA.groupby(['key_year', 'ft_catm_industry_name', 'ft_catm_sector_name', 'ft_catm_country_code', 'ft_catm_region',
                                   'ft_catm_sub_region']).median().reset_index()  #.fillna("NA")

    df_all = df_imputed_data.merge(
        data_to_impute.filter(regex="^(ft_|key_)", axis=1),
        on=["key_ticker", "key_year"],
        how="right",
        suffixes=(None, "_DROP"),
    ).merge(
        df,
        on=["key_ticker"],
        how="left",
        suffixes=(None, "_DROP"),
    )

    df_all = df_all.drop(columns=df_all.filter(regex="(_DROP$)", axis=1).columns)

    dateformat = 'T%Y%m%d'
    all_data_features = [
        CompressedCSVSaver().set_time(time.strftime(dateformat)).set_extension(".tar.gz").set_name("p_combined").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CompressedCSVSaver().set_time(time.strftime(dateformat)).set_extension(".tar.gz").set_name("p_combined").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_companies").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_financials").set_object(df_fin).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_stats").set_object(df_scope_stats).set_datatarget(
            LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_companies").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_financials").set_object(df_fin).set_datatarget(S3Destination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_stats").set_object(df_scope_stats).set_datatarget(
            S3Destination(path="model-data/output")),
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
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_imputations").set_object(df_imputed_data).set_datatarget(
            LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_imputations").set_object(df_imputed_data).set_datatarget(
            S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime(dateformat)).set_name("p_scope_imputations").set_object(df_imputed_data).set_datatarget(
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
