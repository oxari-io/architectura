from base.dataset_loader import PartialLoader
from typing_extensions import Self
from datasources.online import OnlineExcelDatasource
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

