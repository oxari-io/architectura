import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from base import (OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.helper import LogTargetScaler
from datasources.core import PreviousScopeFeaturesDataManager, get_default_datamanager_configuration
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

DATA_DIR = pathlib.Path('local/data')
from lar_calculator.lar_model import OxariUnboundLAR

N_TRIALS = 5
N_STARTUP_TRIALS = 5

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')
    dataset = get_default_datamanager_configuration().add_loader(NetZeroIndexLoader()).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    X = dataset.get_features(OxariDataManager.ORIGINAL)
    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    SPLIT_2 = bag.scope_2
    SPLIT_3 = bag.scope_3

    keys = {
        "key_isin": "text",
        "company_name": "text",
        "ft_catm_industry_name": "text",
        "ft_catm_sector": "text",
        "ticker": "text",
        "country_name": "text",
        "ft_catm_region": "text",
        "ft_catm_sub_region": "text"
    }

    options = {
        "weights": {
            "key_isin": 10,
            "company_name": 5,
            "ticker": 10,
        },
        "name": "TextIndex"
    }

    # Data prepared for saving
    cmb_ld = dataset.categorical_loader
    for ld in dataset.loaders:
        cmb_ld = cmb_ld + ld

    df = dataset.categorical_loader._data.merge(cmb_ld.data, on="key_isin", how="left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df[df.select_dtypes('object').columns] = df[df.select_dtypes('object').columns].fillna("NA")
    df[df.select_dtypes('category').columns] = df[df.select_dtypes('category').columns].astype(str).replace("nan", "NA")

    df_fin = dataset.financial_loader._data

    df_statistics = dataset.scope_loader._data.merge(df, on="key_isin").filter(regex="^(ft|tg|key)", axis=1).drop([
        'key_isin',
        'key_country_code',
        'ft_catm_near_target_status',
        'ft_catm_near_target_year',
        'ft_catm_orga_type',
        'ft_catm_near_target_class',
        'ft_catm_orga_type',
        'ft_catb_committed',
    ],axis=1)
    df_scope_stats = df_statistics.groupby(df_statistics.select_dtypes('object').columns.tolist() + ['key_year']).mean().reset_index().fillna("NA")

    dateformat = 'T%Y%m%d'
    all_data_features = [
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_companies").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_financials").set_object(df_fin).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_scope_stats").set_object(df_scope_stats).set_datatarget(
            LocalDestination(path="model-data/output")),
        # CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_companies").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        # CSVSaver().set_time(time.strftime(dateformat)).set_extension(".csv").set_name("p_financials").set_object(df_fin).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime(dateformat)
                              ).set_name("p_companies").set_object(df).set_datatarget(MongoDestination(index=keys, path="model-data/output", options=options)),
        MongoSaver().set_time(time.strftime(dateformat)).set_name("p_financials").set_object(df_fin).set_datatarget(
            MongoDestination(index={
                "key_isin": ASCENDING,
                "key_year": ASCENDING
            }, path="model-data/output")),
        MongoSaver().set_time(time.strftime(dateformat)).set_name("p_scope_stats").set_object(df_scope_stats).set_datatarget(
            MongoDestination(index={
                "key_year": ASCENDING,
                "ft_catm_industry": ASCENDING,
                "ft_catm_region": ASCENDING,
                "ft_catm_country_code": ASCENDING,
            },
            path="model-data/output")),
    ]

    SavingManager = OxariSavingManager(*all_data_features, )
    SavingManager.run()