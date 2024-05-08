from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from base.constants import DATA_DIR
from base.dataset_loader import (CategoricalLoader, CompanyDataFilter, Datasource, FinancialLoader, OxariDataManager, PartialLoader, ScopeLoader)
from datasources.helper.dedup_func import load_exchange_ranking, name_and_exchange_priority_based_deduplication
from datasources.loaders import RegionLoader
from datasources.local import LocalDatasource
from datasources.online import CachingS3Datasource, S3Datasource


class DefaultDataManager(OxariDataManager):
    # TODO: Follow loader structure of special loaders.
    # TODO: Remove named attributes and pass everything as a list of loaders.
    # TODO: Test if all combinations of loaders work (exclude standard loaders)
    # TODO: Introduce another file which has all the ISIN-YEAR keys
    def __init__(self, *loaders: PartialLoader, verbose=False, **kwargs):
        super().__init__(
            *loaders,
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
        self.logger.info("Taking all previous year scopes")
        df: pd.DataFrame = df.groupby('key_ticker', group_keys=False).progress_apply(self._take_previous_scopes)
        return super()._transform(df)


class ExchangeBasedDeduplicatedScopeDataManager(DefaultDataManager):
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Deduplicating scopes based on exchange priority list")
        self.exch_ranking_path = (Path(__file__).parent / 'misc' / 'exchange_ranking.json').absolute().as_posix()
        df_ = name_and_exchange_priority_based_deduplication(df, self.exch_ranking_path)
        return super()._transform(df_)


class ExchangeBasedDeduplicatedPreviousScopeFeaturesDataManager(PreviousScopeFeaturesDataManager):
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Deduplicating scopes based on exchange priority list")
        self.exch_ranking_path = (Path(__file__).parent / 'misc' / 'exchange_ranking.json').absolute().as_posix()
        df_ = name_and_exchange_priority_based_deduplication(df, self.exch_ranking_path)
        return super()._transform(df_)



class CurrentYearFeatureDataManager(DefaultDataManager):

    def _transform(self, df: pd.DataFrame):
        self.logger.info("Taking current year as feature")
        df["ft_numd_year"] = df["key_year"]
        return super()._transform(df)

class TemporalFeaturesDataManager(PreviousScopeFeaturesDataManager, CurrentYearFeatureDataManager):
    def _transform(self, df: pd.DataFrame):
        self.logger.info("Running two data managers")
        df = super(PreviousScopeFeaturesDataManager, self)._transform(df)
        df = super(CurrentYearFeatureDataManager, self)._transform(df)
        return super()._transform(df)
