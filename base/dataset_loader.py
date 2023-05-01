from __future__ import annotations

import abc
from typing import Dict, List
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Self

from base import OxariLoggerMixin, OxariMixin
from base.common import OxariTransformer
from base.oxari_types import ArrayLike


class Datasource(OxariLoggerMixin, abc.ABC):
    KEYS: List[str] = None
    _COLS: List[str] = None
    _data: pd.DataFrame = None

    def __init__(self, path: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path

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

    def fetch(self, **kwargs) -> Self:
        # https://docs.digitalocean.com/reference/api/spaces-api/
        self.logger.info(f"Fetching data from {self.path}")
        self._check_if_data_exists()
        self._load(**kwargs)
        return self

    @property
    def data(self) -> pd.DataFrame:
        return self._data


class PartialLoader(OxariLoggerMixin, abc.ABC):
    PATTERN = ""
    COL_MAPPING = {}
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

    def __add__(self, other: PartialLoader):
        return CombinedLoader().merge(self, other)

    def __len__(self):
        return len(self.data) if self._data else 0

    @property
    def name(self):
        return self.__class__.__name__

    def set_name(self, name) -> Self:
        self._name = name
        return self

    def set_datasource(self, datasource: Datasource) -> Self:
        self.datasource: Datasource = datasource
        return self

    def load(self, **kwargs) -> Self:
        # TODO: Add caching here! 
        # 1. create .caching folder if not exist
        # 2. specify standard file name for loader based on loader name
        # 3. save loaded data after load function
        # 4. on subsquent load check if file exits locally and load if it does
        # 5. on error save what ever was successfully loaded to caching folder 
        self.logger.info(f'Loading...')
        stime = time.time()
        self._load(**kwargs)
        self.time = time.time() - stime
        self.logger.info(f'Completed download -- {self.time} seconds')
        return self

    def _load(self, **kwargs) -> Self:
        self._data = self.datasource.fetch().data
        return self


class SpecialLoader(PartialLoader):

    @abc.abstractproperty
    def rkeys(self):
        pass

    @abc.abstractproperty
    def lkeys(self):
        pass


class EmptyLoader(PartialLoader):

    def __init__(self, datasource: Datasource = None, verbose=False, **kwargs) -> None:
        super().__init__(None, verbose, **kwargs)
        self._data = pd.DataFrame()

    def __add__(self, other: PartialLoader):

        return CombinedLoader(other.data).set_name(other.name)


class CombinedLoader(PartialLoader):

    def __init__(self, _data: pd.DataFrame = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = _data

    def merge(self, loader_1: PartialLoader = None, loader_2: PartialLoader = None):
        tmp_data: pd.DataFrame = None
        self.logger.info(f"Adding ({loader_1.name} + {loader_2.name})")
        self._name = f"{loader_1.name}-{loader_2.name} "
        if isinstance(loader_2, SpecialLoader):
            self.logger.info(f"Merging special loader {loader_2.name} to {loader_1.name}")
            special_loader: SpecialLoader = loader_2
            tmp_data = loader_1.data.merge(special_loader.data, left_on=special_loader.lkeys, right_on=special_loader.rkeys, how="left").sort_values(special_loader.lkeys)
        if not isinstance(loader_2, SpecialLoader):
            common_keys = list(set(loader_1.keys).intersection(loader_2.keys))
            tmp_data = loader_1.data.merge(loader_2.data, on=common_keys, how="left").sort_values(common_keys)
        self._data = tmp_data
        return self

    @property
    def name(self):
        return self._name.strip()


class OldScopeLoader(PartialLoader):
    PATTERN = "tg_num"

    def __init__(self, threshold=5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

    def _load(self) -> Self:
        # before logging some scopes have very small values so we discard them
        _data = self.datasource.fetch().data
        num_inititial = _data.shape[0]
        threshold = self.threshold
        tmp_cols = [col for col in _data.columns if col.startswith(self.PATTERN)]
        tmp_keys = [col for col in _data.columns if col.startswith("key_")]
        # dropping data entries where unlogged scopes are lower than threshold
        _data[tmp_cols] = np.where((_data[tmp_cols] < threshold), np.nan, _data[tmp_cols])
        # dropping datapoints that have no scopes
        _data = _data.dropna(how="all", subset=tmp_cols)
        _data = _data.dropna(how="any", subset=tmp_keys)

        num_remaining = _data.shape[0]
        self.logger.info(f"From {num_inititial} initial data points removed {num_inititial - num_remaining} data points.")
        self._data = _data
        return self

    @property
    def data(self):
        return self._data


class ScopeLoader(OldScopeLoader):

    def _load(self) -> Self:
        # TODO: before logging some scopes have very small values so we discard them.
        _data = self.datasource.fetch().data
        self._data = _data
        return self


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
        non_features: List[str],
        split_size_val=0.2,
        split_size_test=0.2,
    ) -> None:
        self.data = data
        self.non_features = non_features
        self.split_size_val = split_size_val
        self.split_size_test = split_size_test
        # self.core = self._helper(scope_features)

    @property
    def scope_1(self):
        scope_col = "tg_numc_scope_1"
        return self._helper(scope_col)

    @property
    def scope_2(self):
        scope_col = "tg_numc_scope_2"
        return self._helper(scope_col)

    @property
    def scope_3(self):
        scope_col = "tg_numc_scope_3"
        return self._helper(scope_col)

    def _helper(self, scope_col):
        columns = self.data.columns.difference(self.non_features)
        X = self.data.dropna(how="all", subset=scope_col).copy()
        X = X[X[scope_col] > 0]
        return SplitBag(X[columns], X[scope_col], self.split_size_test, self.split_size_test)


class DataFilter(OxariTransformer):

    def __init__(self, frac: float = 0.1, name=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.frac = frac

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        return super().fit(X, y, **kwargs)

    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        return X


class SimpleDataFilter(DataFilter):

    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        X_new = X.sample(frac=self.frac)
        self.logger.info(f'Filtered dataset from {len(X)} to {len(X_new)} data points')
        return X_new

class CompanyDataFilter(DataFilter):
    def __init__(self, frac: float = 0.1, drop_single_rows=False, name=None, **kwargs) -> None:
        super().__init__(frac, name, **kwargs)
        self.drop_single_rows=drop_single_rows

    def transform(self, X: ArrayLike, **kwargs) -> ArrayLike:
        if self.drop_single_rows:
            group_counts = X.groupby('key_isin').size()
            X = X.groupby('key_isin').filter(lambda x: group_counts[x.name] > 1)
        isins = X["key_isin"].unique()
        self.num_companies_pre = len(isins)
        isin_subset = pd.Series(isins).sample(frac=self.frac).values
        X_new = X[X["key_isin"].isin(isin_subset)]
        self.num_companies_post = len(X_new["key_isin"].unique())
        self.logger.debug(f'Filtered dataset from {self.num_companies_pre} to {self.num_companies_post} companies')
        self.logger.info(f'Filtered dataset from {len(X)} to {len(X_new)} data points')
        return X_new


class OxariDataManager(OxariMixin):
    """
    Handles loading the dataset and keeps versions of each dataset throughout the pipeline.
    Should be capable of reading the data from csv-file or from database
    """
    FINANCIAL = 'financials'
    CATEGORICAL = 'categoricals'
    SCOPE = 'scopes'
    MERGED = 'merged'
    ORIGINAL = 'original'
    IMPUTED_SCOPES = 'imputed_scopes'
    JUMP_RATES = 'jump_rates'
    JUMP_RATES_AGG = 'jump_rates_aggregated'
    IMPUTED_LARS = 'imputed_lars'
    SHORTENED = 'shortened'
    SEARCH_DB = 'search_db'
    REDUCED = 'reduced'

    # NON_FEATURES = ["isin", "year"] + ScopeLoader._COLS
    INDEPENDENT_VARIABLES = []

    def __init__(
            self,
            scope_loader: PartialLoader = None,
            financial_loader: PartialLoader = None,
            categorical_loader: PartialLoader = None,
            other_loaders: List[PartialLoader] = [],
            data_filter: DataFilter = DataFilter(),
            verbose=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.scope_loader = scope_loader
        # self.scope_transformer = scope_transformer or LogarithmScaler(scope_features=self.DEPENDENT_FEATURES)
        self.financial_loader = financial_loader
        self.categorical_loader = categorical_loader
        self.other_loaders = other_loaders
        self.data_filter = data_filter
        self.verbose = verbose
        self._dataset_stack = []
        self.threshold = kwargs.pop("threshold", 5)

    def run(self, **kwargs) -> Self:
        main_loaders = [self.financial_loader, self.scope_loader, self.categorical_loader]

        merged_loader = EmptyLoader()
        self.logger.info(f"Remaining data points {len(merged_loader.data)}")
        for idx, loader in enumerate(main_loaders + self.other_loaders):
            loaded = loader.load()
            loader_name = f"loader_{loaded.name.lower()}"
            merged_loader += loaded
            self.add_data(loader_name, loader.data, loaded.name)
            self.add_data(f"merge_stage_{idx}", merged_loader.data, merged_loader.name)
            # TODO: take len of loader directly
            self.logger.info(f"Remaining data points {len(merged_loader.data)}")

        _df_merged = self.add_data(OxariDataManager.MERGED, merged_loader.data, "Dataset with all parts merged.")
        _df_reduced = self.add_data(OxariDataManager.REDUCED, self.data_filter.fit_transform(_df_merged), "Dataset with reduced number of rows.")
        _df_original = self.add_data(OxariDataManager.ORIGINAL, self._transform(_df_reduced), "Dataset after transformation changes.")
        self.col_non_features = list(_df_original.columns[~_df_original.columns.str.startswith("ft_")])
        self.col_targets = list(_df_original.columns[_df_original.columns.str.startswith("tg_numc_")])
        self.col_features = list(_df_original.columns[_df_original.columns.str.startswith("ft_")])
        self.col_keys = list(_df_original.columns[_df_original.columns.str.startswith("key_")])
        self.col_others = list(_df_original.columns.difference(self.col_targets).difference(self.col_features))
        return self

    # def _reduced(self, df, **kwargs):
    #     num_inititial = df.shape[0]
    #     tmp_targets = [col for col in df.columns if col.startswith("tg_")]
    #     # tmp_keys = [col for col in df.columns if col.startswith("key_")]
    #     # dropping datapoints that have no scopes
    #     df = df.dropna(how="all", subset=tmp_targets)
    #     # dropping all data points with leaky keys
    #     # df = df.dropna(how="any", subset=tmp_keys)

    #     num_remaining = df.shape[0]
    #     self.logger.info(f"From {num_inititial} initial data points removed {num_inititial - num_remaining} data points.")
    #     return df

    #TODO: JUST OVERWRITE THIS ONE
    def _transform(self, df, **kwargs):
        return df.drop_duplicates(['key_isin', 'key_year'])

    def add_data(self, name: str, df: pd.DataFrame, descr: str = "") -> pd.DataFrame:
        self.logger.info(f"Added {name} to {self.__class__.__name__}")
        self._dataset_stack.append((name, df, descr))
        return df

    @property
    def data(self) -> pd.DataFrame:
        return self._dataset_stack[-1][1].copy()

    def get_data_by_name(self, name: str, scope=None) -> pd.DataFrame:
        for nm, df, descr in self._dataset_stack:
            if name == nm:
                df: pd.DataFrame = df.copy().sort_index(axis=1)
                self.logger.info(f"Data with {name} found retrieved: {descr}")
                return df if not scope else df.dropna(subset=scope, how="all")
        self.logger.warn(f"Data with {name} was not found in the dataset-stack")
        return None

    def get_data_by_index(self, index: int) -> pd.DataFrame:
        return self._dataset_stack[index][1].copy()

    def get_data(self, name: str, scope=None):
        data = self.get_data_by_name(name, scope)
        features = data.columns.difference(self.col_non_features)
        X, Y = data[features].copy(), data[scope].copy()
        return X, Y

    def get_scopes(self, name: str):
        data = self.get_data_by_name(name)[self.col_non_features].copy()
        results = data.dropna(how="all", subset=self.col_targets)
        return results

    def get_features(self, name: str):
        data = self.get_data_by_name(name)[self.col_features].copy()
        results = data.dropna(how="all", subset=self.col_features)
        return results

    def get_split_data(self, name: str, split_size_val=0.2, split_size_test=0.2):
        data = self.get_data_by_name(name)
        return SplitScopeDataset(data, self.col_non_features, split_size_val, split_size_test)

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

    def set_filter(self, filter: DataFilter) -> Self:
        self.data_filter = filter
        return self