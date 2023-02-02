from __future__ import annotations

from typing import Dict

import pandas as pd

from base.constants import DATA_DIR
from base.dataset_loader import (CategoricalLoader, Datasource,
                                 FinancialLoader, OxariDataManager,
                                 PartialLoader, ScopeLoader)
from datasources.local import LocalDatasource


class DefaultDataManager(OxariDataManager):

    def __init__(self,
                 scope_loader: Datasource = LocalDatasource(path=DATA_DIR / "scopes_auto.csv"),
                 financial_loader: Datasource = LocalDatasource(path=DATA_DIR / "financials_auto.csv"),
                 categorical_loader: Datasource = LocalDatasource(path=DATA_DIR / "categoricals_auto.csv"),
                 other_loaders: Dict[str, PartialLoader] = {},
                 verbose=False,
                 **kwargs):
        super().__init__(
            scope_loader=ScopeLoader(datasource=scope_loader),
            financial_loader=FinancialLoader(datasource=financial_loader),
            categorical_loader=CategoricalLoader(datasource=categorical_loader),
            other_loaders=other_loaders,
            verbose=verbose,
            **kwargs,
        )


class FSExperimentDataLoader(OxariDataManager):
    # loads the data just like CSVDataLoader, but a selection of the data
    def __init__(self,
                 scope_loader: Datasource = LocalDatasource(path=DATA_DIR / "scopes_auto.csv"),
                 financial_loader: Datasource = LocalDatasource(path=DATA_DIR / "financials_auto.csv"),
                 categorical_loader: Datasource = LocalDatasource(path=DATA_DIR / "categoricals_auto.csv"),
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

    # TODO ask why this run function is different from the run function of OxariDataManager.
    # if answered remove this function because the super funtion is fine
    # def run(self, **kwargs) -> "OxariDataManager":
    #     print("running shortened dataset function")
    #     self.scope_loader = self.scope_loader.run()
    #     self.financial_loader = self.financial_loader.run()
    #     self.categorical_loader = self.categorical_loader.run()
    #     _df_original = self.scope_loader.data.merge(self.financial_loader.data, on=["isin", "year"], how="inner").sort_values(["isin", "year"])
    #     _df_original = _df_original.merge(self.categorical_loader.data, on="isin", how="left")
    #     self.add_data(OxariDataManager.SHORTENED, _df_original, "Dataset without changes.")
    #     return self

    def _transform(self, df, **kwargs):
        # we don't want sampling of the same row more than once
        df_reduced = df.sample(n=5000, replace=False, random_state=1)
        return df_reduced


class PreviousScopeFeaturesDataManager(DefaultDataManager):

    def _take_previous_scopes(self, df: pd.DataFrame):
        df_tmp = df[self.scope_loader._COLS].shift(1)
        df_tmp.columns = [f"ft_numc_preyear_{col}" for col in df_tmp.columns]
        df[df_tmp.columns] = df_tmp
        return df

    def _transform(self, df: pd.DataFrame):
        key_cols = df.columns.filter(regex='^key_',axis=1)
        df = df.sort_values(key_cols, ascending=[True, True]).groupby('key_isin').apply(self._take_previous_scopes)
        return df
