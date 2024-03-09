import io
from pathlib import Path
import pytest
from base.constants import DATA_DIR
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, DataFilter, Datasource, FinancialLoader, OxariDataManager, PartialLoader, ScopeLoader, SimpleDataFilter
from base.run_utils import get_small_datamanager_configuration, get_default_datamanager_configuration

from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
from datasources.loaders import NetZeroIndexLoader, RegionLoader
from datasources.local import LocalDatasource
from datasources.online import CachingS3Datasource, OnlineCSVDatasource, OnlineExcelDatasource, S3Datasource
import pandas as pd
import logging
import os

from tests.fixtures import const_dataset_filtered, const_dataset_full, const_data_manager, const_base_loaders

# logging.basicConfig(level=logging.DEBUG)
# mylogger = logging.getLogger()


@pytest.mark.parametrize("datasource", [
    LocalDatasource(path=DATA_DIR / "scopes.csv"),
    S3Datasource(path="model-data/input/scopes.csv"),
    OnlineExcelDatasource(path="https://sciencebasedtargets.org/download/excel"),
    OnlineCSVDatasource(path="https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv")
])
def test_datasources(datasource: Datasource):
    loaded = datasource.fetch()
    assert len(loaded.data) > 0


@pytest.mark.parametrize("datasource", [CachingS3Datasource(path="model-data/input/file_for_automated_testing.csv")])
def test_datasources_not_cached_download(datasource: Datasource):
    local_path = Path(datasource.path)
    if local_path.exists():
        os.remove(local_path)
    assert not local_path.exists(), "File was not deleted"
    assert not datasource.is_fresh_download, "File was downloaded"
    loaded = datasource.fetch()
    assert datasource.is_fresh_download, "File was not downloaded"
    assert len(loaded.data) > 0, "File does not contain data"
    assert local_path.exists(), "File does not exist locally"
    os.remove(local_path)


@pytest.mark.parametrize("datasource", [CachingS3Datasource(path="model-data/input/file_for_automated_testing.csv.csv")])
def test_datasources_cached_download(datasource: Datasource):
    local_path = Path(datasource.path)
    if not local_path.exists():
        with io.open(local_path, 'w') as f:
            f.write("col1, col2\nd1,d2")

    loaded = datasource.fetch()
    assert len(loaded.data) > 0
    assert not datasource.is_fresh_download
    os.remove(local_path)

@pytest.mark.parametrize("loader", [
    ScopeLoader(datasource=LocalDatasource(path=DATA_DIR / "scopes.csv")),
    FinancialLoader(datasource=LocalDatasource(path=DATA_DIR / "financials.csv")),
    CategoricalLoader(datasource=LocalDatasource(path=DATA_DIR / "categoricals.csv")),
])
def test_standard_loaders(loader: PartialLoader):
    loaded = loader.load()
    assert len(loaded.data) > 0


@pytest.mark.parametrize("loaders", [
    [RegionLoader()],
    [NetZeroIndexLoader()],
    [RegionLoader(), NetZeroIndexLoader()],
])
def test_additional_loaders(loaders: list[PartialLoader], const_base_loaders: list[PartialLoader]):
    
    dataset = DefaultDataManager(*const_base_loaders, *loaders).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    assert len(DATA) > 0


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
])
def test_data_manager(data_manager: OxariDataManager, const_base_loaders:list[PartialLoader]):
    dataset = data_manager.set_filter(CompanyDataFilter()).set_loaders(const_base_loaders).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    assert len(DATA)