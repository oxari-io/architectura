from pathlib import Path
from typing import Dict
from base.dataset_loader import CategoricalLoader, LocalDatasourceMixin, OxariDataManager, PartialLoader, ScopeLoader, FinancialLoader
from base.mappings import CatMapping, NumMapping
import pandas as pd
import numpy as np
import random
from base.constants import DATA_DIR

COLS_CATEGORICALS = CatMapping.get_features()
COLS_FINANCIALS = NumMapping.get_features()

# source: https://cmdlinetips.com/2022/07/randomly-sample-rows-from-a-big-csv-file/
# 21584 is the number of rows in the scope dataset
def sample_n_from_csv(filename:str, n:int=100, total_rows:int=21584) -> pd.DataFrame:
    if total_rows==None:
        with open(filename,"r") as fh:
            total_rows = sum(1 for row in fh)
            print("total rows: ", total_rows)
    
    if(n>total_rows):
        print("Error: n > total_rows") 
    
    skip_rows =  random.sample(range(1,total_rows+1), total_rows-n)
    return pd.read_csv(filename, skiprows=skip_rows)

class CSVScopeLoader(ScopeLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")

        print("")

    def run(self) -> "ScopeLoader":
        self.data = pd.read_csv(self.path)
        self.data = self._clean_up_targets(data = self.data, threshold = self.threshold)
        return self

class FSExperimentCSVScopeLoader(ScopeLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")

    def run(self) -> "ScopeLoader":
        self.data = sample_n_from_csv(self.path, n=500)
        self.data = self._clean_up_targets(data = self.data, threshold = self.threshold)
        return self


class CSVFinancialLoader(FinancialLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "financials.csv")

    def run(self) -> "FinancialLoader":
        super()._check_if_data_exists()
        self.data = pd.read_csv(self.path)
        self.data = self.data[["isin", "year"] + COLS_FINANCIALS]
        return self

class FSExperimentCSVFinancialLoader(FinancialLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "financials.csv")

    def run(self) -> "FinancialLoader":
        super()._check_if_data_exists()
        self.data = sample_n_from_csv(self.path, n=500)
        self.data = self.data[["isin", "year"] + COLS_FINANCIALS]
        return self


class CSVCategoricalLoader(CategoricalLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")

    def run(self) -> "CategoricalLoader":
        super()._check_if_data_exists()
        self.data = pd.read_csv(self.path)
        self.data = self.data[["isin"] + COLS_CATEGORICALS]
        return self

class FSExperimentCSVCategoricalLoader(CategoricalLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")

    def run(self) -> "CategoricalLoader":
        super()._check_if_data_exists()
        self.data = sample_n_from_csv(self.path, n=500)
        self.data = self.data[["isin"] + COLS_CATEGORICALS]
        return self


class CSVDataLoader(OxariDataManager):
    def __init__(self,
                #  object_filename,
                 scope_loader: ScopeLoader = CSVScopeLoader(path  = DATA_DIR / "scopes.csv"),
                 financial_loader: FinancialLoader = CSVFinancialLoader(path = DATA_DIR / "financials.csv"),
                 categorical_loader: CategoricalLoader = CSVCategoricalLoader(path =  DATA_DIR / "categoricals.csv"),
                 other_loaders: Dict[str, PartialLoader] = {},
                 verbose=False,
                 **kwargs):
        super().__init__(
            # object_filename,
            scope_loader,
            financial_loader,
            categorical_loader,
            other_loaders,
            verbose,
            **kwargs,
        )

class FSExperimentDataLoader(OxariDataManager):
    def __init__(self,
                #  object_filename,
                scope_loader: ScopeLoader = FSExperimentCSVScopeLoader(path  = DATA_DIR / "scopes.csv"),
                financial_loader: FinancialLoader = FSExperimentCSVFinancialLoader(path = DATA_DIR / "financials.csv"),
                categorical_loader: CategoricalLoader = FSExperimentCSVCategoricalLoader(path =  DATA_DIR / "categoricals.csv"),
                #  other_loaders: Dict[str, PartialLoader] = {},
                #  verbose=False,
                 **kwargs):
        super().__init__(
            # object_filename,
            scope_loader,
            financial_loader,
            categorical_loader,
            # other_loaders,
            # verbose,
            **kwargs,
        )