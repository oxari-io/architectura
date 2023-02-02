from pathlib import Path
from typing import Dict, List
from base.dataset_loader import CategoricalLoader, LocalDatasource, OxariDataManager, PartialLoader, ScopeLoader, FinancialLoader, S3Datasource, CombinedLoader
from dataset_loader.csv_loader import CSVCategoricalLoader, CSVFinancialLoader
from base.mappings import CatMapping, NumMapping
import pandas as pd
import numpy as np
import random
from base.constants import DATA_DIR
from typing import Generic, TypeVar
from typing_extensions import Self
import abc
from base import OxariMixin
import boto3
from os import environ as env

COLS_CATEGORICALS = CatMapping.get_features()
COLS_FINANCIALS = NumMapping.get_features()


class Datasource(abc.ABC):
    KEYS: List[str] = None
    _COLS: List[str] = None
    _data: pd.DataFrame = None

    @abc.abstractmethod
    def _check_if_data_exists(self, **kwargs) -> bool:
        """
        Implementation should check if self.path is set and then check the specifics for it.
        """
        return None

    @abc.abstractmethod
    def _load(self, **kwargs) -> Self:
        """
        Reads the files and combines them to one single file.
        """
        return self

    def load(self, **kwargs) -> Self:
        # https://docs.digitalocean.com/reference/api/spaces-api/
        self._check_if_data_exists()
        self._load(**kwargs)
        return self

    @property
    def data(self) -> pd.DataFrame:
        return self._data


# T = TypeVar('T', bound=Datasource)


class PartialLoader(abc.ABC):
    PATTERN = ""

    def __init__(self, datasource: Datasource = None, verbose=False, **kwargs) -> None:
        super().__init__()
        self.verbose = verbose
        self._data: pd.DataFrame = None
        self.kwargs = kwargs
        self.set_datasource(datasource)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def columns(self) -> List[str]:
        return [col for col in self._data.columns if (col.startswith(self.PATTERN) or col.startswith("key_"))]

    @property
    def keys(self):
        return [col for col in self._data.columns if (col.startswith("key_"))]

    # TODO: Add in new
    def __add__(self, other: PartialLoader):
        return CombinedLoader(self, other, **self.kwargs)

    # TODO: Add in new
    @property
    def name(self):
        return self.__class__.__name__

    def set_datasource(self, datasource: Datasource) -> Self:
        self.datasource: Datasource = datasource
        return self

    def load(self) -> Self:
        self._data = self.datasource.load().data
        return self


class CombinedLoader(PartialLoader):

    def __init__(self, loader_1: PartialLoader, loader_2: PartialLoader, **kwargs) -> None:
        super().__init__(**kwargs)
        self._name = f"{loader_1.name} + {loader_2.name} "
        common_keys = list(set(loader_1.keys).intersection(loader_2.keys))
        self._data = loader_1.data.merge(loader_2.data, on=common_keys, how="inner").sort_values(common_keys)


class ScopeLoader(OxariMixin, PartialLoader):
    PATTERN = "tg_num"

    def __init__(self, threshold=5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

    @property
    def data(self):
        # before logging some scopes have very small values so we discard them

        data = self.load()._data
        num_inititial = data.shape[0]

        threshold = self.threshold
        tmp_cols = [col for col in self.columns if col.startswith('tg_')]
        # dropping data entries where unlogged scopes are lower than threshold
        data[tmp_cols] = np.where((data[tmp_cols] < threshold), np.nan, data[tmp_cols])
        # dropping datapoints that have no scopes
        data = data.dropna(how="all", subset=self.columns)

        num_remaining = data.shape[0]
        self.logger.info(
            f"From {num_inititial} initial data points, {num_remaining} are complete data points and {num_inititial - num_remaining} data points have missing or invalid scopes")
        result_data = data

        return result_data


class FinancialLoader(PartialLoader):
    PATTERN = "ft_num"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def data(self):
        return self._data[self.columns]


class CategoricalLoader(PartialLoader):
    KEYS = ["isin"]
    PATTERN = "ft_cat"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def data(self):
        return self._data[self.columns]


class LocalDatasource(Datasource):

    def __init__(self, path: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)

    def _check_if_data_exists(self):
        if not self.path.exists():
            self.logger.error(f"Exception: Path(s) does not exist! Got {self.path}")
            raise Exception(f"Path(s) does not exist! Got {self.path}")

    def _load(self) -> "CategoricalLoader":
        self._data = pd.read_csv(self.path)
        return self


class S3Datasource(Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = ''
        self.do_spaces_endpoint = env.get('S3_ENDPOINT')
        self.do_spaces_folder = env.get('S3_BUCKET')
        self.do_spaces_key_id = env.get('S3_KEY_ID')
        self.do_spaces_access_key = env.get('S3_ACCESS_KEY')
        self.do_spaces_region = env.get('S3_REGION')
        self.connect()

    def _check_if_data_exists(self) -> bool:
        self.meta = self.client.head_object(Bucket=self.do_spaces_folder, Key=self.path)

    def _load(self) -> Self:
        # https://docs.digitalocean.com/reference/api/spaces-api/
        response = self.client.get_object(Bucket=self.do_spaces_folder, Key=self.path)
        self._data = pd.read_csv(response['Body'])
        return self

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

class AutoDiscoveryDataManager(OxariDataManager):

    def __init__(self,
                 scope_loader: Datasource = None,
                 financial_loader: Datasource = None,
                 categorical_loader: Datasource = None,
                 other_loaders: Dict[str, PartialLoader] = {},
                 verbose=False,
                 **kwargs):
        super().__init__(
            scope_loader=ScopeLoader(datasource=scope_loader or LocalDatasource(path=DATA_DIR / "scopes_auto.csv")),
            financial_loader=FinancialLoader(datasource=financial_loader or LocalDatasource(path=DATA_DIR / "financials_auto.csv")),
            categorical_loader=CategoricalLoader(datasource=categorical_loader or LocalDatasource(path=DATA_DIR / "categoricals_auto.csv")),
            other_loaders=other_loaders,
            verbose=verbose,
            **kwargs,
        )
