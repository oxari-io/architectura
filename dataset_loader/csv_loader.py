from pathlib import Path
from typing import Dict
from base.dataset_loader import CategoricalLoader, LocalDatasourceMixin, OxariDataManager, PartialLoader, ScopeLoader, FinancialLoader
from base.mappings import CatMapping, NumMapping
import pandas as pd
import numpy as np
from base.constants import DATA_DIR
import boto3
from botocore.client import Config
import io
COLS_CATEGORICALS = CatMapping.get_features()
COLS_FINANCIALS = NumMapping.get_features()


class CSVScopeLoader(ScopeLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")

        print("")

    def run(self) -> "ScopeLoader":
        self.data = pd.read_csv(self.path)
        self.data = self._clean_up_targets(data=self.data, threshold=self.threshold)
        return self


class CSVFinancialLoader(FinancialLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "financials.csv")

    def run(self) -> "FinancialLoader":
        super()._check_if_data_exists()
        self.data = pd.read_csv(self.path)
        self.data = self.data[["isin", "year"] + COLS_FINANCIALS]
        return self


class CSVCategoricalLoader(CategoricalLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")

    def run(self) -> "CategoricalLoader":
        super()._check_if_data_exists()
        self.data = pd.read_csv(self.path)
        self.data = self.data[["isin"] + COLS_CATEGORICALS]
        return self


class S3ScopeLoader(ScopeLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")

        print("")

    def run(self) -> "ScopeLoader":
        # https://docs.digitalocean.com/reference/api/spaces-api/
        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name='nyc3',
            endpoint_url='https://nyc3.digitaloceanspaces.com',
            aws_access_key_id='532SZONTQ6ALKBCU94OU',
            aws_secret_access_key='zCkY83KVDXD8u83RouEYPKEm/dhPSPB45XsfnWj8fxQ',
        )
        f = io.BytesIO()
        client.download_fileobj('BUCKET_NAME', 'OBJECT_NAME', f)        
        self.data = pd.read_csv(self.path)
        self.data = self._clean_up_targets(data=self.data, threshold=self.threshold)
        return self


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
