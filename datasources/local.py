from pathlib import Path
from typing import Dict, List
from base.dataset_loader import CategoricalLoader, Datasource,  OxariDataManager, PartialLoader, ScopeLoader, FinancialLoader,  CombinedLoader
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
import botocore
from os import environ as env



class LocalDatasource(Datasource):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(self.path)

    def _check_if_data_exists(self):
        if not self.path.exists():
            self.logger.error(f"Exception: Path(s) does not exist! Got {self.path}")
            raise Exception(f"Path(s) does not exist! Got {self.path}")

    def _load(self) -> Self:
        self._data = pd.read_csv(self.path)
        return self


class ReducedCSVDatasource(LocalDatasource):
    def _load(self) -> Self:
        super()._load()
        self._data = self._data.sample(0.25)
        return self



