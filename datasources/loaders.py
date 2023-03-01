from base.dataset_loader import PartialLoader, SpecialLoader
from typing_extensions import Self
from datasources.online import OnlineCSVDatasource, OnlineExcelDatasource
import pandas as pd 
import numpy as np

class NetZeroIndexLoader(PartialLoader):
    NEAR_TARGET_YEAR = "ft_catm_near_target_year"
    NEAR_TARGET_CLASS = "ft_catm_near_target_class"
    COL_MAPPING = {
        "ISIN":"key_isin",
        "Near term - Target Status":"ft_catm_near_target_status",
        "Near term - Target Classification":NEAR_TARGET_CLASS,
        "Near term - Target Year":NEAR_TARGET_YEAR,
        "Net-Zero Committed":"ft_catb_committed",
        "Organization Type":"ft_catm_orga_type",
    }
    def __init__(self, **kwargs) -> None:
        datasource = kwargs.pop('datasource', OnlineExcelDatasource(path="https://sciencebasedtargets.org/download/excel"))
        super().__init__(datasource=datasource, **kwargs)


    def _load(self) -> Self:
        _data = self.datasource.fetch().data
        _data = _data.rename(columns=self.COL_MAPPING)[self.COL_MAPPING.values()]
        _data[self.NEAR_TARGET_YEAR] = _data[self.NEAR_TARGET_YEAR].fillna('9999')
        # Removing annoying string characters 
        _data[self.NEAR_TARGET_YEAR] = _data[self.NEAR_TARGET_YEAR].astype(str).str.replace("FY", "").str.replace("Y", "").str.replace("F", "").str.replace("/", ",")
        # Taking max element of list entries like 2030,2024,2030 
        _data[self.NEAR_TARGET_YEAR] = _data[self.NEAR_TARGET_YEAR].str.split(",").apply(lambda elems: max([int(e) for e in elems]))
        # Assign to groups
        _data[self.NEAR_TARGET_YEAR] = pd.cut(_data[self.NEAR_TARGET_YEAR], bins=[0, 2020, 2030, 2050, 2100], right=True)
        _data[self.NEAR_TARGET_CLASS] = _data[self.NEAR_TARGET_CLASS].fillna('NA').str.split("/").apply(lambda elems: elems[-1])
        self._data = _data
        return self

class RegionLoader(SpecialLoader):
    RKEY = "key_country_code"
    LKEY = "ft_catm_country_code"

    COL_MAPPING = {
        "alpha-3":RKEY,
        "region":"ft_catm_region",
        "sub-region":"ft_catm_sub_region",
    }
    def __init__(self, **kwargs) -> None:
        datasource = kwargs.pop('datasource', OnlineCSVDatasource(path="https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"))
        super().__init__(datasource=datasource, **kwargs)


    def _load(self) -> Self:
        _data = self.datasource.fetch().data
        _data = _data.rename(columns=self.COL_MAPPING)[self.COL_MAPPING.values()]
        self._data = _data
        return self

    @property
    def rkeys(self):
        return [self.RKEY]

    @property
    def lkeys(self):
        return [self.LKEY]