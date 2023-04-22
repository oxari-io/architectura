from base.common import OxariMetaModel
import pytest
from base.dataset_loader import Datasource
from datastores.saver import CSVSaver, DataTarget, LocalDestination, MongoDestination, MongoSaver, OxariSavingManager, PartialSaver, PickleSaver, S3Destination
import time
import pandas as pd
from tests.fixtures import const_data_manager, const_pipeline, const_meta_model, const_example_df, const_example_df_multi_rows, const_example_dict, const_example_dict_multi_rows, const_example_series, const_dataset_filtered, const_data_for_scope_imputation

@pytest.mark.parametrize("destination", [
    LocalDestination(path="model-data/output"),
    S3Destination(path="model-data/output"),
])
def test_model_destinations(destination: DataTarget, const_meta_model:OxariMetaModel):
    saver = PickleSaver()
    saver = saver.set_name("t_model")
    saver = saver.set_extension(".pkl")
    saver = saver.set_object(const_meta_model)
    saver = saver.set_datatarget(destination)
    assert saver.save(), f"Saving {const_meta_model} to {destination} failed"

@pytest.mark.parametrize("destination", [
    LocalDestination(path="model-data/output"),
    S3Destination(path="model-data/output"),
])
def test_csv_destinations(destination: DataTarget, const_data_for_scope_imputation:pd.DataFrame):
    saver = CSVSaver()
    saver = saver.set_name("t_scope")
    saver = saver.set_extension(".csv")
    saver = saver.set_object(const_data_for_scope_imputation)
    saver = saver.set_datatarget(destination)
    assert saver.save(), f"Saving data to {destination} failed"

@pytest.mark.parametrize("destination", [
    MongoDestination(path="model-data/output"),
])
def test_mongo_destinations(destination: DataTarget, const_data_for_scope_imputation:pd.DataFrame):
    saver = MongoSaver()
    saver = saver.set_name("t_scope")
    saver = saver.set_object(const_data_for_scope_imputation)
    saver = saver.set_datatarget(destination)
    assert saver.save(), f"Saving data to {destination} failed"




