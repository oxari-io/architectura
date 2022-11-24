import io
import dill as pkl
import cloudpickle

from os import PathLike

from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import numpy as np
import abc
from base.common import OxariMixin
from base.mappings import CatMapping, NumMapping
from sklearn.model_selection import train_test_split
from base import OxariModel, OxariDataManager



class DestinationMixin(abc.ABC):
    @abc.abstractmethod
    def _check_if_destination_accessible(self) -> bool:
        """
        Implementation should check if self.path is set and then check the specifics for it.
        """
        return None


class LocalDestinationMixin(DestinationMixin):
    def __init__(self, local_path: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.local_path = Path(local_path)
        self._check_if_destination_accessible()

    def _check_if_destination_accessible(self):
        if not self.local_path.exists():
            raise Exception(f"Path(s) does not exist! Got {self.local_path}")


class PartialSaver(abc.ABC):
    def __init__(self, verbose=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self.data: pd.DataFrame = None
        self.columns: List[str] = None

    # @abc.abstractmethod
    # def run(self) -> "PartialSaver":
    #     """
    #     Reads the files and combines them to one single file.
    #     """
    #     return self


class OxariSavingManager(LocalDestinationMixin):
    """
    Saves the files into the appropriate location
    """
 
    def __init__(
        self,
        # object_filename,
        meta_model: OxariModel = None,
        dataset: OxariDataManager = None,
        # lar_model: CategoricalLoader = None,
        other_savers: Dict[str, PartialSaver] = None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.meta_model = meta_model
        self.dataset = dataset
        # self.other_savers = categorical_loader
        # self.object_filename = object_filename
        self.verbose = verbose
        # self._dataset_stack = []
        # self._df_original: pd.DataFrame = None
        # self._df_preprocessed: pd.DataFrame = None
        # self._df_filled: pd.DataFrame = None
        # self._df_estimated: pd.DataFrame = None
        # self.threshold = kwargs.pop("threshold", 5)

    def run(self, **kwargs) -> "OxariDataManager":
        pass

    def save_model_locally(self, today):
        self._check_if_destination_accessible()

        with open(self.local_path / f"MetaModel_{today}.p", "wb") as handle:
            cloudpickle.dump(self.meta_model, handle)

        # pkl.dump(self.meta_model, io.open(self.local_path / f"MetaModel_{today}.pkl", 'wb'))

