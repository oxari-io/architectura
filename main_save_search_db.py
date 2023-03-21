import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from base import (OxariDataManager, OxariMetaModel, helper)
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.helper import LogTargetScaler
from datasources.core import PreviousScopeFeaturesDataManager
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
from pymongo import TEXT

DATA_DIR = pathlib.Path('local/data')
from lar_calculator.lar_model import OxariUnboundLAR

N_TRIALS = 5
N_STARTUP_TRIALS = 5

if __name__ == "__main__":
    today = time.strftime('%d-%m-%Y')

    dataset = PreviousScopeFeaturesDataManager(S3Datasource(path='model-input-data/scopes_auto.csv'),
                                               S3Datasource(path='model-input-data/financials_auto.csv'),
                                               S3Datasource(path='model-input-data/categoricals_auto.csv'),
                                               other_loaders=[RegionLoader(), NetZeroIndexLoader()]).run()
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
    for ld in dataset.other_loaders:
        cmb_ld = cmb_ld + ld

    df = dataset.categorical_loader._data.merge(cmb_ld.data, on="key_isin", how="left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df[df.select_dtypes('object').columns] = df[df.select_dtypes('object').columns].fillna("NA")
    df[df.select_dtypes('category').columns] = df[df.select_dtypes('category').columns].astype(str)

    df_fin = dataset.financial_loader._data

    all_data_features = [
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("p_companies").set_object(df).set_datatarget(LocalDestination(path="model-data/output")),
        CSVSaver().set_time(time.strftime('%d-%m-%Y')).set_name("p_companies").set_object(df).set_datatarget(S3Destination(path="model-data/output")),
        MongoSaver().set_time(time.strftime('%d-%m-%Y')).set_name("p_companies").set_object(df).set_datatarget(MongoDestination(index=keys, path="model-data/output")),
        MongoSaver().set_time(time.strftime('%d-%m-%Y')).set_name("p_financials").set_object(df_fin).set_datatarget(MongoDestination(index=keys, path="model-data/output")),
    ]

    SavingManager = OxariSavingManager(*all_data_features, )
    SavingManager.run()