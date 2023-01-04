from pathlib import Path
from typing import Dict
from base.dataset_loader import CategoricalLoader, LocalDatasourceMixin, OxariDataManager, PartialLoader, ScopeLoader, FinancialLoader
from base.mappings import CatMapping, NumMapping
import pandas as pd
import numpy as np
from base.constants import DATA_DIR

COLS_CATEGORICALS = CatMapping.get_features()
COLS_FINANCIALS = NumMapping.get_features()


class CSVScopeLoader(ScopeLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")

        print("")

    def run(self) -> "ScopeLoader":
        self.data = pd.read_csv(self.path)
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


class CSVCategoricalLoader(CategoricalLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")

    def run(self) -> "CategoricalLoader":
        super()._check_if_data_exists()
        self.data = pd.read_csv(self.path)
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

class FeatureSelectionExperimentDataLoader(OxariDataManager):
    pass