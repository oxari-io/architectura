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
from base import OxariModel, OxariDataManager


class DestinationMixin(abc.ABC):
    @abc.abstractmethod
    def _check_if_destination_accessible(self) -> bool:
        """
        Implementation should check if self.path is set and then check the specifics for it.
        """
        return None


class LocalDestinationMixin(DestinationMixin):
    def __init__(self, path: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self._check_if_destination_accessible()

    def _check_if_destination_accessible(self):
        if not self.path.exists():
            raise Exception(f"Path(s) does not exist! Got {self.path}")


class PartialSaver(abc.ABC):
    def __init__(self, verbose=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self.data: pd.DataFrame = None
        self.columns: List[str] = None

    @abc.abstractmethod
    def run(self) -> "PartialSaver":
        """
        Reads the files and combines them to one single file.
        """
        return self




# TODO: Finish this saving manager. Take the timestamp into account when deciding on a name. 
class OxariSavingManager(OxariMixin, abc.ABC):
    """
    Handles loading the dataset and keeps versions of each dataset throughout the pipeline.
    Should be capable of reading the data from csv-file or from database
    """
 
    def __init__(
        self,
        # object_filename,
        meta_model: OxariModel = None,
        dataset: OxariDataManager = None,
        # lar_model: CategoricalLoader = None,
        other_savers: Dict[str, PartialLoader] = None,
        verbose=False,
        **kwargs,
    ):
        self.meta_model = scope_loader
        self.dataset = financial_loader
        self.other_savers = categorical_loader
        # self.object_filename = object_filename
        self.verbose = verbose
        self._dataset_stack = []
        self._df_original: pd.DataFrame = None
        self._df_preprocessed: pd.DataFrame = None
        self._df_filled: pd.DataFrame = None
        self._df_estimated: pd.DataFrame = None
        self.threshold = kwargs.pop("threshold", 5)

    def run(self, **kwargs) -> "OxariDataManager":
        pass
