from pathlib import Path
from typing import Dict
from base.dataset_loader import CategoricalLoader, LocalDatasource, OxariDataManager, PartialLoader, ScopeLoader, FinancialLoader, Datasource
from base.mappings import CatMapping, NumMapping
import pandas as pd
import numpy as np
from base.constants import DATA_DIR
import boto3
# from botocore import client
from os import environ as env
import io

COLS_CATEGORICALS = CatMapping.get_features()
COLS_FINANCIALS = NumMapping.get_features()


class CSVScopeLoader(ScopeLoader, LocalDatasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")


class CSVFinancialLoader(FinancialLoader, LocalDatasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "financials.csv")


class CSVCategoricalLoader(CategoricalLoader, LocalDatasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")


class S3Datasource(Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = ''
        self.do_spaces_endpoint = env.get('S3_ENDPOINT')
        self.do_spaces_folder = env.get('S3_BUCKET')
        self.do_spaces_key_id = env.get('S3_KEY_ID')
        self.do_spaces_access_key = env.get('S3_ACCESS_KEY')
        self.do_spaces_region = env.get('S3_REGION')

    def _check_if_data_exists(self) -> bool:
        self.meta = self.client.head_object(Bucket=self.do_spaces_folder, Key=self.file_name)

    def connect(self):
        self.session: boto3.Session = boto3.Session()
        self.client = self.session.client(
            's3',
            region_name=self.do_spaces_region,
            endpoint_url=f'{self.do_spaces_endpoint}',
            aws_access_key_id=self.do_spaces_key_id,
            aws_secret_access_key=self.do_spaces_access_key,
        )
        return self.client

    # TODO: change to load instead of run
    def run(self) -> "S3Datasource":
        # https://docs.digitalocean.com/reference/api/spaces-api/
        self._check_if_data_exists()
        response = self.client.get_object(Bucket=self.do_spaces_folder, Key=self.file_name)
        self._data = pd.read_csv(response['Body'])
        return self


class S3ScopeLoader(ScopeLoader, S3Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = "scopes.csv"
        self.client = self.connect()


class S3FinancialLoader(FinancialLoader, S3Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = "financials.csv"
        self.client = self.connect()


class S3CategoricalLoader(CategoricalLoader, S3Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = "categoricals.csv"
        self.client = self.connect()


class CSVDataManager(OxariDataManager):

    def __init__(self,
                 scope_loader: ScopeLoader = CSVScopeLoader(path=DATA_DIR / "scopes.csv"),
                 financial_loader: FinancialLoader = CSVFinancialLoader(path=DATA_DIR / "financials.csv"),
                 categorical_loader: CategoricalLoader = CSVCategoricalLoader(path=DATA_DIR / "categoricals.csv"),
                 other_loaders: Dict[str, PartialLoader] = {},
                 verbose=False,
                 **kwargs):
        super().__init__(
            scope_loader,
            financial_loader,
            categorical_loader,
            other_loaders,
            verbose,
            **kwargs,
        )


# class DigitalOceanSpacesDataManager(CSVDataManager):
