import io
import cloudpickle as pkl
import base
import pipeline
import preprocessors
import scope_estimators
import postprocessors
import imputers
import feature_reducers

MODULES_TO_PICKLE = [
    base,
    pipeline,
    preprocessors,
    scope_estimators,
    postprocessors,
    imputers,
    feature_reducers,
]

from os import PathLike

from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import numpy as np
import abc
from sklearn.model_selection import train_test_split
from base import OxariModel, OxariDataManager




class LocalDestinationMixin():
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
        self._store = None

    @abc.abstractmethod
    def save(self, **kwargs) -> "PartialSaver":
        return self
    
    @abc.abstractmethod
    def _check_if_destination_accessible(self) -> bool:
        """
        Implementation should check if self.path is set and then check the specifics for it.
        """
        return True    




class MetaModelSaver(PartialSaver, abc.ABC):
    def __init__(self, model: OxariModel):
        self._store = model

class DataSaver(PartialSaver, abc.ABC):
    def __init__(self, data: OxariModel):
        self._store = data

class LARModelSaver(PartialSaver, abc.ABC):
    def __init__(self, model: OxariModel):
        self._store = model


class LocalModelSaver(MetaModelSaver, LocalDestinationMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> "LocalModelSaver":
        super()._check_if_data_exists()
        return self

    def save(self, today, name="", **kwargs):
        self._check_if_destination_accessible()
        name = "noname" if name == "" else name
        pkl.dump(self._store, io.open(self.local_path / f"MetaModel_{today}_{name}.pkl", 'wb'))


class OxariSavingManager():
    """
    Saves the files into the appropriate location
    """
    def __init__(
        self,
        # object_filename,
        meta_model: MetaModelSaver = None,  
        dataset: DataSaver = None,
        lar_model: LARModelSaver = None,
        other_savers: Dict[str, PartialSaver] = None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.saver_meta_model = meta_model
        self.saver_dataset = dataset
        self.saver_lar_model = lar_model
        # self.other_savers = categorical_loader
        self.verbose = verbose
        self._register_all_modules_to_pickle()

    def run(self, **kwargs) -> "OxariDataManager":
        self.saver_meta_model.save(**kwargs)

    def _register_all_modules_to_pickle(self):
        for md in MODULES_TO_PICKLE:
            pkl.register_pickle_by_value(md)
