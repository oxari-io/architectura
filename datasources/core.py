from __future__ import annotations

from typing import Dict, List

import pandas as pd

from base.constants import DATA_DIR
from base.dataset_loader import (CategoricalLoader, Datasource,
                                 FinancialLoader, OxariDataManager,
                                 PartialLoader, ScopeLoader)
from datasources.local import LocalDatasource


class DefaultDataManager(OxariDataManager):
    # TODO: Follow loader structure of special loaders. 
    # TODO: Remove named attributes and pass everything as a list of loaders.
    # TODO: Test if all combinations of loaders work (exclude standard loaders)
    # TODO: Introduce another file which has all the ISIN-YEAR keys
    def __init__(self,
                 scope_loader: Datasource = LocalDatasource(path=DATA_DIR / "scopes_auto.csv"),
                 financial_loader: Datasource = LocalDatasource(path=DATA_DIR / "financials_auto.csv"),
                 categorical_loader: Datasource = LocalDatasource(path=DATA_DIR / "categoricals_auto.csv"),
                 other_loaders: List[PartialLoader] = [],
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


class FSExperimentDataLoader(DefaultDataManager):

    def _transform(self, df, **kwargs):
        # we don't want sampling of the same row more than once
        df_reduced = df.sample(n=5000, replace=False, random_state=1)
        return df_reduced


class PreviousScopeFeaturesDataManager(DefaultDataManager):
    PREFIX = "ft_numc_prior_"
    def _take_previous_scopes(self, df: pd.DataFrame):
        df_tmp = df.iloc[:, df.columns.str.startswith('tg_numc_')].shift(1)
        df_tmp.columns = [f"{self.PREFIX}{col}" for col in df_tmp.columns]
        df[df_tmp.columns] = df_tmp
        return df

    def _transform(self, df: pd.DataFrame):
        key_cols = list(df.columns[df.columns.str.startswith('key')])
        df = df.sort_values(key_cols, ascending=True).groupby('key_isin', group_keys=False).apply(self._take_previous_scopes)
        return super()._transform(df)
