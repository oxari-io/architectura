import pytest
from base.constants import DATA_DIR
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, DataFilter, Datasource, FinancialLoader, OxariDataManager, PartialLoader, ScopeLoader, SimpleDataFilter

from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
from datasources.loaders import NetZeroIndexLoader, RegionLoader
from datasources.local import LocalDatasource
from datasources.online import OnlineCSVDatasource, OnlineExcelDatasource, S3Datasource
import pandas as pd
import logging

from tests.fixtures import const_dataset_filtered, const_dataset_full, const_data_manager

# logging.basicConfig(level=logging.DEBUG)
# mylogger = logging.getLogger()


@pytest.mark.parametrize("datasource", [
    LocalDatasource(path=DATA_DIR / "scopes_auto.csv"),
    S3Datasource(path="model-input-data/scopes_auto.csv"),
    OnlineExcelDatasource(path="https://sciencebasedtargets.org/download/excel"),
    OnlineCSVDatasource(path="https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv")
])
def test_datasources(datasource: Datasource):
    loaded = datasource.fetch()
    assert len(loaded.data) > 0


@pytest.mark.parametrize("loader", [
    ScopeLoader(datasource=LocalDatasource(path=DATA_DIR / "scopes_auto.csv")),
    FinancialLoader(datasource=LocalDatasource(path=DATA_DIR / "financials_auto.csv")),
    CategoricalLoader(datasource=LocalDatasource(path=DATA_DIR / "categoricals_auto.csv")),
])
def test_standard_loaders(loader: PartialLoader):
    loaded = loader.load()
    assert len(loaded.data) > 0


@pytest.mark.parametrize("loaders", [
    [RegionLoader()],
    [NetZeroIndexLoader()],
    [RegionLoader(), NetZeroIndexLoader()],
])
def test_additional_loaders(loaders: list[PartialLoader]):
    dataset = DefaultDataManager(other_loaders=loaders).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    assert len(DATA) > 0
    for l in loaders:
        for col in l.COL_MAPPING.values():
            assert col in DATA


@pytest.mark.parametrize("filter", [
    CompanyDataFilter(0.1),
    SimpleDataFilter(0.1),
])
def test_filters(filter: DataFilter, const_dataset_full: pd.DataFrame):
    loaded = filter.fit_transform(const_dataset_full)
    assert len(loaded) < len(const_dataset_full)


def test_splits(const_data_manager: OxariDataManager):
    BAGSIZE = 2
    bag = const_data_manager.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    assert hasattr(SPLIT_1, "train")
    assert hasattr(SPLIT_1, "rem")
    assert hasattr(SPLIT_1, "val")
    assert hasattr(SPLIT_1, "test")

    SPLIT_1 = bag.scope_1
    assert len(SPLIT_1.train) == BAGSIZE
    SPLIT_2 = bag.scope_2
    assert len(SPLIT_2.train) == BAGSIZE
    SPLIT_3 = bag.scope_3
    assert len(SPLIT_3.train) == BAGSIZE


@pytest.mark.parametrize("data_manager", [
    DefaultDataManager(),
    PreviousScopeFeaturesDataManager(),
    PreviousScopeFeaturesDataManager(other_loaders=[RegionLoader(), NetZeroIndexLoader()]),
])
def test_data_manager(data_manager: OxariDataManager):
    dataset = data_manager.set_filter(CompanyDataFilter()).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    assert len(DATA)