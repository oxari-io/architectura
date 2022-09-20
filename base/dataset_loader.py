from os import PathLike
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import abc
from base.common import OxariMixin


class BaseLoader(OxariMixin):
    """
    Handles loading the dataset and keeps versions of each dataset throughout the pipeline.
    Should be capable of reading the data from csv-file or from database
    """
    def __init__(self, ) -> None:
        self.path: Path = None
        self._df_original: pd.DataFrame = None
        self._df_preprocessed: pd.DataFrame = None
        self._df_filled: pd.DataFrame = None
        self._df_estimated: pd.DataFrame = None

    def set_path(self, path: Union[str, PathLike] = None) -> "BaseLoader":
        self.path = Path(path)
        return self

    def read_data(self) -> "BaseLoader":
        if not self.path.exists():
            raise Exception(f"Path '{self.path}' does not exist")
        self._df_original = pd.DataFrame(self.path)
        return self

    def set_original_data(self, df: pd.DataFrame) -> "BaseLoader":
        self._df_original = df
        return self

    def set_preprocessed_data(self, df: pd.DataFrame) -> "BaseLoader":
        self._df_preprocessed = df
        return self

    def set_filled_data(self, df: pd.DataFrame) -> "BaseLoader":
        self._df_filled = df
        return self

    def set_estimated(self, df: pd.DataFrame) -> "BaseLoader":
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

    @abc.abstractmethod
    def train_test_val_split(self, split_size_test: float, split_size_val: float) -> "BaseLoader":
        """
        Creating targets(y) and features(X) from pandas.DataFrame
        Split the data with sklearn train_test_split() and returns 3 subsets: training, testing, and validation;
        Parameters:
        data (pandas.DataFrame): pre-processed dataset in pandas format from data pipeline
        split_size_test (float): ratio (from 0 to 1) of the splitting training-testing dataset
        split_size_val (float): ratio (from 0 to 1) of the splitting training-val dataset
        Returns:
        numpy array: arrays for X (train, test, val) and 3 for y (train, test, val)
        """
        return self