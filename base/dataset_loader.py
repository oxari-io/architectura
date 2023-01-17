from __future__ import annotations
from os import PathLike
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import numpy as np
import abc
from base import OxariMixin, OxariTransformer
# from base.common import OxariMixin
from base.mappings import CatMapping, NumMapping
from sklearn.model_selection import train_test_split
from base.helper import LogarithmScaler
from base.oxari_types import ArrayLike, Self
from collections import namedtuple
import boto3
from os import environ as env
import io


class Datasource(abc.ABC):
    KEYS: List[str] = None
    _COLS: List[str] = None
    data: pd.DataFrame = None

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


class S3Datasource(Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = ''
        self.do_spaces_endpoint = env.get('S3_ENDPOINT')
        self.do_spaces_folder = env.get('S3_BUCKET')
        self.do_spaces_key_id = env.get('S3_KEY_ID')
        self.do_spaces_access_key = env.get('S3_ACCESS_KEY')
        self.do_spaces_region = env.get('S3_REGION')
        self.connect()

    def _check_if_data_exists(self) -> bool:
        self.meta = self.client.head_object(Bucket=self.do_spaces_folder, Key=self.file_name)

    def _load(self) -> Self:
        # https://docs.digitalocean.com/reference/api/spaces-api/
        response = self.client.get_object(Bucket=self.do_spaces_folder, Key=self.file_name)
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


class LocalDatasource(Datasource):

    def __init__(self, path: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self._check_if_data_exists()

    def _check_if_data_exists(self):
        if not self.path.exists():
            raise Exception(f"Path(s) does not exist! Got {self.path}")

    def _load(self) -> "CategoricalLoader":
        self._data = pd.read_csv(self.path)
        return self


class PartialLoader(abc.ABC):

    def __init__(self, verbose=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self._data: pd.DataFrame = None
        self.columns: List[str] = None

    @property
    def data(self):
        return self._data


class ScopeLoader(OxariMixin, PartialLoader, abc.ABC):
    KEYS = ["isin", "year"]
    _COLS = NumMapping.get_targets()

    def __init__(self, threshold=5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.columns = [f"{col}" for col in ScopeLoader._COLS]

    @property
    def data(self):
        # before logging some scopes have very small values so we discard them

        data = self._data
        num_inititial = data.shape[0]

        threshold = self.threshold

        # dropping data entries where unlogged scopes are lower than threshold
        data[self.columns] = np.where((data[self.columns] < threshold), np.nan, data[self.columns])
        # dropping datapoints that have no scopes
        data = data.dropna(how="all", subset=self.columns)

        if self.verbose:
            num_remaining = data.shape[0]
            print(
                f"*** From {num_inititial} initial data points, {num_remaining} are complete data points and {num_inititial - num_remaining} data points have missing or invalid scopes ***"
            )
            
        self.logger.info(f"*** From {num_inititial} initial data points, {num_remaining} are complete data points and {num_inititial - num_remaining} data points have missing or invalid scopes ***")
        result_data = data

        return result_data


class FinancialLoader(PartialLoader, abc.ABC):
    KEYS = ["isin", "year"]
    _COLS = NumMapping.get_features()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.columns = [f"ft_fin_{col}" for col in FinancialLoader._COLS]

    @property
    def data(self):
        result_data = self._data[self.KEYS + self._COLS]
        result_data[self.columns] = result_data[self._COLS]
        result_data = result_data.drop(self._COLS, axis=1)
        return result_data


class CategoricalLoader(PartialLoader, abc.ABC):
    KEYS = ["isin"]
    _COLS = CatMapping.get_features()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.columns = [f"ft_cat_{col}" for col in CategoricalLoader._COLS]

    @property
    def data(self):
        result_data = self._data[self.KEYS + self._COLS]
        result_data[self.columns] = result_data[self._COLS]
        result_data = result_data.drop(self._COLS, axis=1)
        return result_data


class SplitBag():

    class Pair:

        def __init__(self, X, y) -> None:
            self.X = X
            self.y = y
            self._iter = [self.X, self.y]

        def __len__(self):
            return 2

        def __iter__(self):
            return (i for i in self._iter)

        def __getitem__(self, key):
            return self._iter[key]

    def __init__(self, X: ArrayLike, y: ArrayLike, split_size_val=0.2, split_size_test=0.2, **kwargs) -> None:
        X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=split_size_test)
        X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, test_size=split_size_val)
        self.rem = SplitBag.Pair(X_rem, y_rem)
        self.train = SplitBag.Pair(X_train, y_train)
        self.val = SplitBag.Pair(X_val, y_val)
        self.test = SplitBag.Pair(X_test, y_test)


class SplitScopeDataset():

    def __init__(
        self,
        data: ArrayLike,
        scope_features: List[str],
        split_size_val=0.2,
        split_size_test=0.2,
    ) -> None:
        self.data = data
        self.scope_features = scope_features
        self.split_size_val = split_size_val
        self.split_size_test = split_size_test
        # self.core = self._helper(scope_features)

    @property
    def scope_1(self):
        scope_col = "scope_1"
        return self._helper(scope_col)

    @property
    def scope_2(self):
        scope_col = "scope_2"
        return self._helper(scope_col)

    @property
    def scope_3(self):
        scope_col = "scope_3"
        return self._helper(scope_col)

    def _helper(self, scope_col):
        columns = self.data.columns.difference(self.scope_features)
        X = self.data.dropna(how="all", subset=scope_col).copy()
        return SplitBag(X[columns], X[scope_col], self.split_size_test, self.split_size_test)


HOW_TO_MERGE = "inner"


class OxariDataManager(OxariMixin):
    """
    Handles loading the dataset and keeps versions of each dataset throughout the pipeline.
    Should be capable of reading the data from csv-file or from database
    """
    ORIGINAL = 'original'
    FINANCIAL = 'financials'
    CATEGORICAL = 'categoricals'
    SCOPE = 'scopes'
    MERGED = 'merged'
    ORIGINAL = 'original'
    IMPUTED_SCOPES = 'imputed_scopes'
    JUMP_RATES = 'jump_rates'
    JUMP_RATES_AGG = 'jump_rates_aggregated'
    IMPUTED_LARS = 'imputed_lars'

    # NON_FEATURES = ["isin", "year"] + ScopeLoader._COLS
    DEPENDENT_VARIABLES = ScopeLoader._COLS
    INDEPENDENT_VARIABLES = []

    def __init__(
        self,
        scope_loader: ScopeLoader = None,
        financial_loader: FinancialLoader = None,
        categorical_loader: CategoricalLoader = None,
        other_loaders: Dict[str, Datasource] = None,
        verbose=False,
        **kwargs,
    ):
        self.scope_loader = scope_loader
        # self.scope_transformer = scope_transformer or LogarithmScaler(scope_features=self.DEPENDENT_FEATURES)
        self.financial_loader = financial_loader
        self.categorical_loader = categorical_loader
        self.other_loaders = other_loaders
        self.verbose = verbose
        self._dataset_stack = []
        self.threshold = kwargs.pop("threshold", 5)

    def run(self, **kwargs) -> "OxariDataManager":
        scope_loader = self.scope_loader.load()
        financial_loader = self.financial_loader.load()
        categorical_loader = self.categorical_loader.load()
        self.non_features = self.scope_loader.KEYS + self.scope_loader.columns
        _df_merged = self.add_data(OxariDataManager.MERGED, self._merge(scope_loader, financial_loader, categorical_loader), "Dataset with all parts merged.")
        _df_original = self.add_data(OxariDataManager.ORIGINAL, self._transform(_df_merged), "Dataset after transformation changes.")
        return self

    def _merge(self, scope_loader:ScopeLoader, financial_loader:FinancialLoader, categorical_loader:CategoricalLoader, **kwargs):
        _df_original: pd.DataFrame = scope_loader.data.dropna(subset="isin")
        _df_original = _df_original.merge(financial_loader.data, on=financial_loader.KEYS, how="inner").sort_values(financial_loader.KEYS)
        _df_original = _df_original.merge(categorical_loader.data, on=categorical_loader.KEYS, how="left")
        return _df_original

    def _transform(self, df, **kwargs):
        return df

    def add_data(self, name: str, df: pd.DataFrame, descr: str = "") -> pd.DataFrame:
        self._dataset_stack.append((name, df, descr))
        return df

    @property
    def data(self) -> pd.DataFrame:
        return self._dataset_stack[-1][1].copy()

    def get_data_by_name(self, name: str, scope=None) -> pd.DataFrame:
        for nm, df, descr in self._dataset_stack:
            if name == nm:
                df: pd.DataFrame = df.copy().sort_index(axis=1)
                return df if not scope else df.dropna(subset=scope, how="all")

    def get_data_by_index(self, index: int) -> pd.DataFrame:
        return self._dataset_stack[index][1].copy()

    def get_data(self, name: str, scope=None):
        data = self.get_data_by_name(name, scope)
        features = data.columns.difference(self.non_features)
        X, Y = data[features].copy(), data[scope].copy()
        return X, Y

    def get_scopes(self, name: str):
        return self.get_data_by_name(name)[self.non_features].copy()

    def get_features(self, name: str):
        data = self.get_data_by_name(name)
        features = data.columns.difference(self.non_features)
        return data[features].copy()

    def get_split_data(self, name: str, split_size_val=0.2, split_size_test=0.2):
        data = self.get_data_by_name(name)
        return SplitScopeDataset(data, self.non_features, split_size_val, split_size_test)

    # where is this called from? Where do we make an API constructor for the object?
    @staticmethod
    def train_test_val_split(X, y, split_size_test, split_size_val):
        """
        Splitting the data in trianing, testing, and validation sets
        with a splitting threshold of split_size_test, and split_size_val respectively

        Parameters:
        data (pandas.DataFrame): pre-processed dataset in pandas format from data pipeline
        split_size_test (float): splitting threshold between training set and testing + validation sets
        split_size_val (float): splitting threshold between testing set and validation set

        Return:
        X_train (numpy array): training data (features)
        y_train (numpy array): training data (targets)
        X_train_full (numpy array): not splitted training data (features)
        y_train_full (numpy array): not splitted training data (targets)
        X_test (numpy array): testing data (features)
        y_test (numpy array): testing data (targets)
        X_val (numpy array): validation data (features)
        y_val (numpy array): validation data (targets)
        """

        #  REVIEWME: is this needed? probably not
        # excluding -1 as they are not valid targets
        # data = data.loc[data[self.scope] != -1]

        # split in features and targets

        selector = ~np.isnan(y)

        # verbose
        # print(f"Number of datapoints: shape of y {y.shape}, shape of X {X.shape}")
        OxariDataManager.logger.debug(f"Number of datapoints: shape of y {y.shape}, shape of X {X.shape}")
        
        X_rem, X_test, y_rem, y_test = train_test_split(X[selector], y[selector], test_size=split_size_test)

        # splitting further - train and validation sets will be used for optimization; test set will be used for performance assesment
        X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, test_size=split_size_val)

        # return X_train, y_train, X_train_full, y_train_full, X_test, y_test, X_val, y_val
        return X_rem, y_rem, X_train, y_train, X_val, y_val, X_test, y_test
