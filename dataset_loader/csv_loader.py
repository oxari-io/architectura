from pathlib import Path
from typing import Dict
from base.dataset_loader import CategoricalLoader, LocalDatasourceMixin, OxariDataLoader, PartialLoader, ScopeLoader, FinancialLoader
from base.mappings import CatMapping, NumMapping
import pandas as pd
import numpy as np

OBJECT_DIR = Path("local/objects")
DATA_DIR = Path("local/data")
COLS_CATEGORICALS = CatMapping.get_features()


class CSVScopeLoader(ScopeLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")
        super().__init__(**kwargs)

    def run(self) -> "ScopeLoader":
        self.data = pd.read_csv(self.path)
        self.data = self._clean_up_targets(self.data, self.threshold)
        return self


class CSVFinancialLoader(FinancialLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        self.path = kwargs.pop("path", DATA_DIR / "financials.csv")
        super().__init__(**kwargs)

    def run(self) -> "FinancialLoader":
        super()._check_if_data_exists()
        self.data = pd.read_csv(self.path)
        self.data = self.data[COLS_CATEGORICALS + ["isin"]]
        return self


class CSVCategoricalLoader(CategoricalLoader, LocalDatasourceMixin):
    def __init__(self, **kwargs):
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")
        super().__init__(**kwargs)

    def run(self) -> "CategoricalLoader":
        super()._check_if_data_exists()
        self.data = pd.read_csv(self.path)

        return self


class CSVDataLoader(OxariDataLoader):
    def __init__(self,
                 object_filename,
                 scope_loader: ScopeLoader = CSVScopeLoader(),
                 financial_loader: FinancialLoader = CSVFinancialLoader(),
                 categorical_loader: CategoricalLoader = CSVCategoricalLoader(),
                 other_loaders: Dict[str, PartialLoader] = {},
                 verbose=False,
                 **kwargs):
        super().__init__(
            object_filename,
            scope_loader,
            financial_loader,
            categorical_loader,
            other_loaders,
            verbose,
            **kwargs,
        )
