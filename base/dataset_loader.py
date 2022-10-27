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
        # TODO: find a more efficient way to do the following

        data = kwargs.get("data", self.data)
        num_inititial = data.shape[0]

        threshold = kwargs.get("threshold", self.threshold)

        # dropping datapoints that have no scopes
        data = data.dropna(how="all", subset=self.scopes_columns)
        # dropping data entries where unlogged scopes are lower than threshold
        ss = NumMapping.get_targets()
        data[ss].loc[data[ss] < threshold] = np.nan
        # for s in NumMapping.get_targets():
        #     data.loc[data[s] < threshold, [s]] = np.nan

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


class OxariDataLoader(abc.ABC):
    """
    Handles loading the dataset and keeps versions of each dataset throughout the pipeline.
    Should be capable of reading the data from csv-file or from database
    """
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
        self._df_original: pd.DataFrame = None
        self._df_preprocessed: pd.DataFrame = None
        self._df_filled: pd.DataFrame = None
        self._df_estimated: pd.DataFrame = None
        self.threshold = kwargs.pop("threshold", 5)

    def run(self, **kwargs) -> "OxariDataLoader":
        self.scope_loader = self.scope_loader.run()
        self.financial_loader = self.financial_loader.run()
        self.categorical_loader = self.categorical_loader.run()
        # TODO: Think whether this should be called via @property
        self._df_original = self.scope_loader.data.merge(self.financial_loader.data, on="isin", how="left").merge(self.categorical_loader.data, on="isin", how="left")
        return self

    def set_original_data(self, df: pd.DataFrame) -> "OxariDataLoader":
        self._df_original = df
        return self

    def set_preprocessed_data(self, df: pd.DataFrame) -> "OxariDataLoader":
        self._df_preprocessed = df
        return self

    def set_filled_data(self, df: pd.DataFrame) -> "OxariDataLoader":
        self._df_filled = df
        return self

    def set_estimated(self, df: pd.DataFrame) -> "OxariDataLoader":
        self._df_estimated = df
        return self

    @property
    def original_data(self):
        return self._df_original.copy()

    @property
    def preprocessed_data(self):
        return self._df_preprocessed.copy()

    @property
    def filled_data(self):
        return self._df_filled.copy()

    @property
    def estimated_data(self):
        return self._df_estimated.copy()

    # # @abc.abstractmethod
    # def train_test_val_split(self, split_size_test: float, split_size_val: float) -> "OxariDataLoader":
    #     """
    #     Creating targets(y) and features(X) from pandas.DataFrame
    #     Split the data with sklearn train_test_split() and returns 3 subsets: training, testing, and validation;
    #     Parameters:
    #     data (pandas.DataFrame): pre-processed dataset in pandas format from data pipeline
    #     split_size_test (float): ratio (from 0 to 1) of the splitting training-testing dataset
    #     split_size_val (float): ratio (from 0 to 1) of the splitting training-val dataset
    #     Returns:
    #     numpy array: arrays for X (train, test, val) and 3 for y (train, test, val)
    #     """
    #     return self
    
    def train_test_val_split(self, split_size_test, split_size_val, list_of_skipped_columns, scope):
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
        X, y = self._df_filled.drop(columns = list_of_skipped_columns),  self._df_filled[f"group_label_{scope}"]

        # verbose
        print(f"Number of datapoints: shape of y {y.shape}, shape of X {X.shape}")

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=split_size_test)

        # splitting further - train and validation sets will be used for optimization; test set will be used for performance assesment
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=split_size_val)

        return X_train, y_train, X_train_full, y_train_full, X_test, y_test, X_val, y_val
