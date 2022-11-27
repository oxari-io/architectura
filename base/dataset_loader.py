"""
OxariMixin has been removed from the inheritance of the DataLoader class. 
That is because DataLoader does not need methods as set_optimizer or load_state, etc.

Also, object_filename has been removed from DataLoader objects because what's the use of assigning a name to a class which won't be saved as a file.
"""

from os import PathLike
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import numpy as np
import abc
from base.common import OxariMixin
# from base.common import OxariMixin
from base.mappings import CatMapping, NumMapping
from sklearn.model_selection import train_test_split


class DatasourceMixin(abc.ABC):
    @abc.abstractmethod
    def _check_if_data_exists(self) -> bool:
        """
        Implementation should check if self.path is set and then check the specifics for it.
        """
        return None


class LocalDatasourceMixin(DatasourceMixin):
    def __init__(self, path: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self._check_if_data_exists()

    def _check_if_data_exists(self):
        if not self.path.exists():
            raise Exception(f"Path(s) does not exist! Got {self.path}")


class PartialLoader(abc.ABC):
    def __init__(self, verbose=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self.data: pd.DataFrame = None
        self.columns: List[str] = None

    @abc.abstractmethod
    def run(self) -> "PartialLoader":
        """
        Reads the files and combines them to one single file.
        """
        return self


class ScopeLoader(PartialLoader, abc.ABC):
    def __init__(self, threshold=5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.columns = NumMapping.get_targets()

    def _clean_up_targets(self, **kwargs):
        # before logging some scopes have very small values so we discard them

        data = kwargs.get("data", self.data)
        num_inititial = data.shape[0]

        threshold = kwargs.get("threshold", self.threshold)

        # dropping data entries where unlogged scopes are lower than threshold
        data[self.columns] = np.where((data[self.columns] < threshold), np.nan, data[self.columns])
        # dropping datapoints that have no scopes
        data = data.dropna(how="all", subset=self.columns)

        if self.verbose:
            num_remaining = data.shape[0]
            print(
                f"*** From {num_inititial} initial data points, {num_remaining} are complete data points and {num_inititial - num_remaining} data points have missing or invalid scopes ***"
            )
        return data


class FinancialLoader(PartialLoader, abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.columns = NumMapping.get_features()


class CategoricalLoader(PartialLoader, abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.columns = CatMapping.get_features()


class OxariDataManager(OxariMixin):
    """
    Handles loading the dataset and keeps versions of each dataset throughout the pipeline.
    Should be capable of reading the data from csv-file or from database
    """
    ORIGINAL = 'original'
    IMPUTED = 'imputed_scopes'
    
    def __init__(
        self,
        # object_filename,
        scope_loader: ScopeLoader = None,
        financial_loader: FinancialLoader = None,
        categorical_loader: CategoricalLoader = None,
        other_loaders: Dict[str, PartialLoader] = None,
        verbose=False,
        **kwargs,
    ):
        self.scope_loader = scope_loader
        self.financial_loader = financial_loader
        self.categorical_loader = categorical_loader
        self.other_loaders = other_loaders
        # self.object_filename = object_filename
        self.verbose = verbose
        self._dataset_stack = []
        self._df_original: pd.DataFrame = None
        self._df_preprocessed: pd.DataFrame = None
        self._df_filled: pd.DataFrame = None
        self._df_estimated: pd.DataFrame = None
        self.threshold = kwargs.pop("threshold", 5)

    def run(self, **kwargs) -> "OxariDataManager":
        self.scope_loader = self.scope_loader.run()
        self.financial_loader = self.financial_loader.run()
        self.categorical_loader = self.categorical_loader.run()
        _df_original = self.scope_loader.data.merge(self.financial_loader.data, on=["isin", "year"], how="inner").sort_values(["isin", "year"])
        _df_original = _df_original.merge(self.categorical_loader.data, on="isin", how="left")
        # TODO: Use class constant instead of manual string to name dataset versions on OxariDataManager.add_data
        self.add_data(OxariDataManager.ORIGINAL, _df_original, "Dataset without changes.")
        return self

    def add_data(self, name: str, df: pd.DataFrame, descr: str = "") -> "OxariDataManager":
        self._dataset_stack.append((name, df, descr))
        return self

    @property
    def data(self) -> pd.DataFrame:
        return self._dataset_stack[-1][1].copy()

    def get_data_by_name(self, name: str) -> pd.DataFrame:
        for nm, df, descr in self._dataset_stack:
            if name == nm:
                return df.copy()

    def get_data_by_index(self, index: int) -> pd.DataFrame:
        return self._dataset_stack[index][1].copy()

    def get_scopes(self, name: str):
        return self.get_data_by_name(name)[["isin", "year", "scope_1", "scope_2", "scope_3"]]

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
        print(f"Number of datapoints: shape of y {y.shape}, shape of X {X.shape}")

        X_rem, X_test, y_rem, y_test = train_test_split(X[selector], y[selector], test_size=split_size_test)

        # splitting further - train and validation sets will be used for optimization; test set will be used for performance assesment
        X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, test_size=split_size_val)

        # return X_train, y_train, X_train_full, y_train_full, X_test, y_test, X_val, y_val
        return X_rem, y_rem, X_train, y_train, X_val, y_val, X_test, y_test
