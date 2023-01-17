from pathlib import Path
from typing import Dict
from base.dataset_loader import CategoricalLoader, LocalDatasource, OxariDataManager, PartialLoader, ScopeLoader, FinancialLoader, Datasource, S3Datasource
from base.mappings import CatMapping, NumMapping
import pandas as pd
import numpy as np
from base.constants import DATA_DIR


COLS_CATEGORICALS = CatMapping.get_features()
COLS_FINANCIALS = NumMapping.get_features()


class CSVScopeLoader(ScopeLoader, LocalDatasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")


class CSVFinancialLoader(FinancialLoader, LocalDatasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "financials.csv")


class CSVCategoricalLoader(CategoricalLoader, LocalDatasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")



# TODO: The internet loaders need a progressbar
class S3ScopeLoader(ScopeLoader, S3Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = "scopes.csv"
        self.client = self.connect()


class S3FinancialLoader(FinancialLoader, S3Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = "financials.csv"
        self.client = self.connect()


class S3CategoricalLoader(CategoricalLoader, S3Datasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = "categoricals.csv"
        self.client = self.connect()


class DefaultDataManager(OxariDataManager):

    def __init__(self,
                 scope_loader: ScopeLoader = CSVScopeLoader(path=DATA_DIR / "scopes.csv"),
                 financial_loader: FinancialLoader = CSVFinancialLoader(path=DATA_DIR / "financials.csv"),
                 categorical_loader: CategoricalLoader = CSVCategoricalLoader(path=DATA_DIR / "categoricals.csv"),
                 other_loaders: Dict[str, PartialLoader] = {},
                 verbose=False,
                 **kwargs):
        super().__init__(
            scope_loader,
            financial_loader,
            categorical_loader,
            other_loaders,
            verbose,
            **kwargs,
        )

class PreviousScopeFeaturesDataManager(DefaultDataManager):
    def _take_previous_scopes(self, df:pd.DataFrame):
        df_tmp = df[self.scope_loader._COLS].shift(1)
        df_tmp.columns = [f"ft_fin_preyear_{col}" for col in df_tmp.columns]
        df[df_tmp.columns] = df_tmp
        return df
    
    def _transform(self, df:pd.DataFrame):
        df = df.sort_values(['isin', 'year'], ascending=[True, True]).groupby('isin').apply(self._take_previous_scopes)
        return df
