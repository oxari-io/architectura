

import pytest

from base.dataset_loader import CompanyDataFilter, OxariDataManager
from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager


@pytest.fixture
def training_data_filtered():
    dataset = DefaultDataManager().set_filter(CompanyDataFilter(0.1)).run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    return DATA

@pytest.fixture
def training_data_full():
    dataset = DefaultDataManager().run()
    DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
    return DATA