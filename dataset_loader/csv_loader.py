from pathlib import Path
from typing import Dict
from base.dataset_loader import CategoricalLoader, LocalDatasource, OxariDataManager, PartialLoader, ScopeLoader, FinancialLoader, Datasource, S3Datasource
from base.mappings import CatMapping, NumMapping
import pandas as pd
import numpy as np
import random
from base.constants import DATA_DIR


COLS_CATEGORICALS = CatMapping.get_features()
COLS_FINANCIALS = NumMapping.get_features()

# source: https://cmdlinetips.com/2022/07/randomly-sample-rows-from-a-big-csv-file/
# 21584 is the number of rows in the scope dataset ??????
def sample_n_from_csv(filename:str, n:int=100, total_rows:int=21584) -> pd.DataFrame:
    if total_rows==None:
        with open(filename,"r") as fh:
            total_rows = sum(1 for row in fh)
            print("total rows: ", total_rows)
    
    if(n>total_rows):
        print("Error: n > total_rows") 
    
    skip_rows =  random.sample(range(1,total_rows+1), total_rows-n)
    return pd.read_csv(filename, skiprows=skip_rows)

class CSVScopeLoader(ScopeLoader, LocalDatasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")


class CSVFinancialLoader(FinancialLoader, LocalDatasource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "financials.csv")


class FSExperimentCSVScopeLoader(ScopeLoader, LocalDatasource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "scopes.csv")

    # def run(self) -> "ScopeLoader":
    #     self.data = sample_n_from_csv(self.path, n=500)
    #     self.data = self._clean_up_targets(data = self.data, threshold = self.threshold)
    #     return self


class FSExperimentCSVFinancialLoader(FinancialLoader, LocalDatasource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "financials.csv")

    # def run(self) -> "FinancialLoader":
    #     super()._check_if_data_exists()
    #     self.data = sample_n_from_csv(self.path, n=500)
    #     self.data = self.data[["isin", "year"] + COLS_FINANCIALS]
    #     return self

class CSVCategoricalLoader(CategoricalLoader, LocalDatasource):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")


class FSExperimentCSVCategoricalLoader(CategoricalLoader, LocalDatasource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = kwargs.pop("path", DATA_DIR / "categoricals.csv")

    # def run(self) -> "CategoricalLoader":
    #     super()._check_if_data_exists()
    #     self.data = sample_n_from_csv(self.path, n=500)
    #     self.data = self.data[["isin"] + COLS_CATEGORICALS]
    #     return self


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

class FSExperimentDataLoader(OxariDataManager):
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
    def _take_previous_scopes(self, df:pd.DataFrame):
        df_tmp = df[self.scope_loader._COLS].shift(1)
        df_tmp.columns = [f"ft_fin_preyear_{col}" for col in df_tmp.columns]
        df[df_tmp.columns] = df_tmp
        return df
    
    def _transform(self, df:pd.DataFrame):
        df = df.sort_values(['isin', 'year'], ascending=[True, True]).groupby('isin').apply(self._take_previous_scopes)
        return df
