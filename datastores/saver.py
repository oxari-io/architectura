import io

import cloudpickle as pkl
from jmespath.ast import index
from statsmodels.multivariate.factor_rotation import target_rotation

import base
import feature_reducers
import imputers
import lar_calculator
import pipeline
import postprocessors
import preprocessors
import scope_estimators
import tempfile
import tqdm

MODULES_TO_PICKLE = [
    base,
    pipeline,
    preprocessors,
    scope_estimators,
    postprocessors,
    imputers,
    feature_reducers,
    lar_calculator,
]
import abc
import time
from os import PathLike, environ as env
from pathlib import Path
from typing import Dict, List
import pandas as pd
import boto3
import botocore
from mypy_boto3_s3 import S3Client
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from typing_extensions import Self
from base import OxariDataManager, OxariLoggerMixin, OxariMetaModel

ROOT_LOCAL = "local/objects"
ROOT_REMOTE = "remote"


class DataTarget(OxariLoggerMixin, abc.ABC):

    def __init__(self, path: PathLike = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path = path

    @abc.abstractmethod
    def _check_if_destination_accessible(self, **kwargs) -> bool:
        """
        Implementation should check if self.path is set and then check the specifics for it.
        """
        return True

    @abc.abstractmethod
    def _save(self, obj, name, **kwargs) -> bool:
        target_destination = self._path / name
        with io.open(target_destination, "wb") as file:
            obj_to_save = pkl.dumps(obj)
            file.write(obj_to_save)
        return target_destination.absolute()

    # @abc.abstractmethod
    # def _save_fallback(self, **kwargs) -> bool:
    #     return False


class PartialSaver(OxariLoggerMixin, abc.ABC):

    def __init__(self, verbose=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self._store = None

    def set_datatarget(self, datatarget: DataTarget) -> Self:
        self.datatarget: DataTarget = datatarget
        return self

    def set_object(self, obj) -> Self:
        self.object = obj
        return self

    def set_name(self, name: str) -> Self:
        self._name = name
        return self

    def set_path(self, path: str) -> Self:
        self._path = path
        return self

    def set_time(self, time: str) -> Self:
        self._time = time
        return self

    @property
    def name(self):
        return f"{self._name}_{self._time}"

    def save(self, **kwargs) -> bool:
        try:
            self.datatarget._check_if_destination_accessible(**kwargs)
            self.datatarget._save(self.object, self.name, **kwargs)
            return True
        except Exception as e:
            # TODO: Needs local emergency saving in case of exception
            self.logger.error(f"ERROR: Something went horribly wrong while saving '{self._name}': {e}")
            return False


class LocalDestination(DataTarget):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path = Path(self._path) if self._path else Path('.')

    def _check_if_destination_accessible(self):
        if not self._path.exists():
            self.logger.error(f"Exception: Path(s) do/does not exist! Got {self._path.absolute()}")
            raise Exception(f"Path(s) do/does not exist! Got {self._path.absolute()}")

    def _create_path(self):
        self._path.mkdir(parents=True, exist_ok=True)

    def _save(self, obj, name, **kwargs) -> bool:
        return super()._save(obj, name, **kwargs)


class S3Destination(DataTarget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._path = Path(self._path) if self._path else Path('')
        self.do_spaces_key_id = env.get('S3_KEY_ID')
        self.do_spaces_access_key = env.get('S3_ACCESS_KEY')
        self.do_spaces_endpoint = env.get('S3_ENDPOINT')  # Endpoint ${REGION}.digitaloceanspaces.com
        self.do_spaces_bucket = env.get('S3_BUCKET')  # DO-Space
        self.do_spaces_region = env.get('S3_REGION')  # Repetition of ${REGION}
        self.connect()

    def connect(self) -> S3Client:
        self.session: boto3.Session = boto3.Session()
        self.client = self.session.client(
            's3',
            config=botocore.config.Config(s3={'addressing_style': 'virtual'}),
            region_name=self.do_spaces_region,
            endpoint_url=self.do_spaces_endpoint,
            aws_access_key_id=self.do_spaces_key_id,
            aws_secret_access_key=self.do_spaces_access_key,
        )
        return self.client

    def _check_if_destination_accessible(self):
        # TODO: Construct a proper check
        test = self.session.get_available_resources()

    def _save(self, obj, name, **kwargs):
        pkl_stream = pkl.dumps(obj)
        _key = self._path / f"{name}"
        self.client.put_object(Body=pkl_stream, Bucket=self.do_spaces_bucket, Key=_key.as_posix())
        return True

class MongoDestination(DataTarget):

    def __init__(self, index=None, options=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.index=index
        self.options=options
        self._path = Path(self._path) if self._path else Path('.')
        self._connection_string = env.get('MONGO_CONNECTION_STRING')
        self.connect()

    def _check_if_destination_accessible(self):
        if not self._path.exists():
            self.logger.error(f"Exception: Path(s) do/does not exist! Got {self._path.absolute()}")
            raise Exception(f"Path(s) do/does not exist! Got {self._path.absolute()}")

    def connect(self) -> MongoClient:
        self.client = MongoClient(self._connection_string, server_api=ServerApi('1'))
        return self.client

    def _save(self, obj:pd.DataFrame, name, **kwargs) -> bool:
        if not isinstance(obj, pd.DataFrame):
            raise Exception(f'Object is not a dataframe but {obj.__class__}')
        db = self.client['companies']
        
        collection = db[name]
        collection.drop()
        self.logger.info(f"Dropped collection '{db.name}/{collection.name}'")
        inserted = collection.insert_many(obj.to_dict('records'))
        if self.index:
            collection.create_index(name='TextIndex', keys=list(self.index.items()))

        self.logger.info(f"Inserted {len(inserted.inserted_ids)} rows.")
        return True

class PickleSaver(PartialSaver, abc.ABC):

    @property
    def name(self):
        return f"{self._name}_{self._time}.pkl"


class CSVSaver(PartialSaver, abc.ABC):

    @property
    def name(self):
        return f"{self._name}_{self._time}.csv"


class MongoSaver(PartialSaver, abc.ABC):

    @property
    def name(self):
        return f"{self._name}_{self._time}"

# class DataSaver(PartialSaver, abc.ABC):
#     SUB_FOLDER = Path("objects/estimates")

#     def set(self, dataset: OxariDataManager) -> Self:
#         self._store = dataset
#         return self

#     @property
#     def name(self):
#         return f"estimates_{self._name}_{self._time}"

# class LARModelSaver(PartialSaver, abc.ABC):
#     SUB_FOLDER = Path("objects/lar_model")

#     def set(self, model: OxariMetaModel) -> "LARModelSaver":
#         self._store = model
#         return self

#     @property
#     def name(self):
#         return f"lar_model_{self._name}_{self._time}"

# class LocalMetaModelSaver(LocalDestination, BlobSaver):

#     def _save(self, **kwargs) -> bool:

#         return pkl.dump(self._store, io.open(self.destination_path / f"{self.name}.pkl", 'wb'))

# class LocalLARModelSaver(LocalDestination, LARModelSaver):

#     def _save(self, **kwargs) -> bool:
#         return pkl.dump(self._store, io.open(self.destination_path / f"{self.name}.pkl", 'wb'))

# class LocalDataSaver(LocalDestination, DataSaver):

#     def _save(self, **kwargs) -> bool:
#         # TODO: Save all data stored in the manager
#         # TODO: On Exception do fallback
#         csv_name_1 = self.destination_path / f"scope_imputed_{self._time}_{self._name}.csv"
#         csv_name_2 = self.destination_path / f"lar_imputed_{self._time}_{self._name}.csv"
#         scope_imputed = self._store.get_data_by_name(OxariDataManager.IMPUTED_SCOPES)
#         lar_imputed = self._store.get_data_by_name(OxariDataManager.IMPUTED_LARS)
#         scope_imputed.to_csv(csv_name_1, index=False)
#         lar_imputed.to_csv(csv_name_2, index=False)

# class S3MetaModelSaver(S3Destination, BlobSaver):

#     def _save(self, **kwargs):
#         pkl_stream = pkl.dumps(self._store)
#         self.client.put_object(Body=pkl_stream, Bucket='remote', Key=str(self.SUB_FOLDER / f"{self.name}.pkl"))

# class S3LARModelSaver(S3Destination, LARModelSaver):

#     def _save(self, **kwargs):
#         pkl_stream = pkl.dumps(self._store)
#         self.client.put_object(Body=pkl_stream, Bucket='remote', Key=str(self.SUB_FOLDER / f"{self.name}.pkl"))

# class S3DataSaver(S3Destination, DataSaver):
#     def _save(self, **kwargs) -> bool:
#         # TODO: Save all data stored in the manager
#         csv_name_1 = f"scope_imputed_{self.name}"
#         csv_name_2 = f"lar_imputed_{self.name}"
#         scope_imputed = self._store.get_data_by_name(OxariDataManager.IMPUTED_SCOPES)
#         lar_imputed = self._store.get_data_by_name(OxariDataManager.IMPUTED_LARS)
#         self.client.put_object(Body=scope_imputed.to_csv(index=False), Bucket='remote', Key=str(self.SUB_FOLDER / f"{csv_name_1}.csv"))
#         self.client.put_object(Body=lar_imputed.to_csv(index=False), Bucket='remote', Key=str(self.SUB_FOLDER / f"{csv_name_2}.csv"))


class OxariSavingManager(OxariLoggerMixin):
    """
    Saves the files into the appropriate location
    """

    def __init__(
        self,
        *savers: PartialSaver,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.savers = savers
        self.verbose = kwargs.get('verbose')
        self._register_all_modules_to_pickle()

    def run(self, **kwargs) -> Self:
        for saver in tqdm.tqdm(self.savers):
            self.logger.info(f"Saved {saver.name} via {saver.__class__}")
            saver.save(**kwargs)

    def _register_all_modules_to_pickle(self):
        # https://oegedijk.github.io/blog/pickle/dill/python/2020/11/10/serializing-dill-references.html
        # https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs
        for md in MODULES_TO_PICKLE:
            pkl.register_pickle_by_value(md)
