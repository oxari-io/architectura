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
from base.run_utils import compute_jump_rates, compute_lar, impute_missing_years, impute_scopes
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
import os
import pathlib
from pickle import Unpickler

from lar_calculator.lar_model import OxariUnboundLAR

MODEL_OUTPUT_DIR = pathlib.Path('model-data/output')

class CustomUnpickler(Unpickler):

    def find_class(self, module, name):
        if module == 'pathlib':
            if name == 'WindowsPath':
                return pathlib.PosixPath if os.name != 'nt' else pathlib.WindowsPath
            elif name == 'PosixPath':
                return pathlib.WindowsPath if os.name == 'nt' else pathlib.PosixPath
        return super().find_class(module, name)


def custom_loads(pickled_data):
    return CustomUnpickler(pickled_data).load()


# def download_latest_model(model_name):

#     # naming convention for model:
#     session = boto3.session.Session()
#     client = session.client(
#         's3',
#         region_name='ams3',
#         endpoint_url='https://oxari-storage.ams3.digitaloceanspaces.com',
#         aws_access_key_id=DOS_ACCESS_KEY,
#         aws_secret_access_key=DOS_SECRET_KEY)
#     client.download_file('model-data', "output/" + MODEL_NAME,
#                          f'objects/{model_name}')
#     logger.info(f"Downloaded {model_name} successfully.")

#     # client.download_file('model-data', "output/"+MODEL_NAME, f'objects/{model_name}')


def load_model_from_disk(full_path):
    print(full_path)
    with open(full_path, 'rb') as f:
        # pkl_stream = f.read()
        model = custom_loads(f)
        return model



if __name__ == "__main__":
    dummy_features = {
    'ft_catm_industry_name': 'Finance',
    'ft_catm_sector_name': 'I dont know',
    'ft_catm_country_code': 'DEU',
    'ft_numc_cash': 10000.0,
    'ft_numd_employees': 200,
    'ft_numc_equity': 42.0,
    'ft_numc_inventories': 1.0,
    'ft_numc_market_cap': 10.0,
    'ft_numc_net_income': 1.0,
    'ft_numc_ppe': 4.2,
    'ft_numc_rd': 42.0,
    'ft_numc_revenue': 420.0,
    'ft_numc_roa': 42.0,
    'ft_numc_roe': 42.0,
    'ft_numc_stock_return': 10.0,
    'ft_numc_total_assets': 10.0,
    'ft_numc_total_liab': 42.0
    }
    model = load_model_from_disk(MODEL_OUTPUT_DIR / 'T20240318_p_model.pkl')
    print(model.predict(dummy_features))