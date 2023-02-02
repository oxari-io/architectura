import io
import cloudpickle as pkl
import base
import pipeline
import preprocessors
import scope_estimators
import postprocessors
import imputers
import feature_reducers
import lar_calculator

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
import time
from pathlib import Path
from typing import Dict
import abc
from base import OxariMetaModel, OxariDataManager, OxariLoggerMixin
from os import environ as env
import boto3

ROOT_LOCAL = "local"
ROOT_REMOTE = "remote"

class Destination(abc.ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.destination_path = Path('.')


class LocalDestination(Destination):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.destination_path = ROOT_LOCAL / self.SUB_FOLDER

        
    def _check_if_destination_accessible(self):
        if not self.destination_path.exists():
            self.logger.error(f"Exception: Path(s) do/does not exist! Got {self.destination_path.absolute()}")
            raise Exception(f"Path(s) do/does not exist! Got {self.destination_path.absolute()}")

    def _create_path(self):
        self.destination_path.mkdir(parents=True, exist_ok=True)        


class S3Destination(Destination):

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.do_spaces_endpoint = env.get('S3_ENDPOINT')
        self.do_spaces_folder = env.get('S3_BUCKET')
        self.do_spaces_key_id = env.get('S3_KEY_ID')
        self.do_spaces_access_key = env.get('S3_ACCESS_KEY')
        self.do_spaces_region = env.get('S3_REGION')
        self.connect()

    def connect(self):
        self.session: boto3.Session = boto3.Session()
        self.client = self.session.client(
            's3',
            region_name=self.do_spaces_region,
            endpoint_url=f'{self.do_spaces_endpoint}',
            aws_access_key_id=self.do_spaces_key_id,
            aws_secret_access_key=self.do_spaces_access_key,
        )
        return self.client

    def _check_if_destination_accessible(self):
        # TODO: Construct a proper check
        test = self.session.get_available_resources()


class PartialSaver(OxariLoggerMixin, abc.ABC):
    
    def __init__(self, time=time.strftime('%d-%m-%Y'), name="noname", verbose=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self._store = None
        self._time = time
        self._name = name

    @abc.abstractmethod
    def _save(self, **kwargs) -> bool:
        return False

    # @abc.abstractmethod
    # def _save_fallback(self, **kwargs) -> bool:
    #     return False

    def save(self, **kwargs) -> bool:
        try:
            self._check_if_destination_accessible()
            self._save(**kwargs)
            return True
        except Exception as e:
            # TODO: Needs local emergency saving in case of exception
            self.logger.error(f"ERROR: Something went horribly wrong while saving '{self._name}': {e}")
            # print(f"ERROR: Something went horribly wrong while saving '{self._name}': {e}")
            
            return False

    @abc.abstractmethod
    def _check_if_destination_accessible(self) -> bool:
        """
        Implementation should check if self.path is set and then check the specifics for it.
        """
        return True

    @abc.abstractmethod
    def set(self, **kwargs) -> "PartialSaver":
        pass


class MetaModelSaver(PartialSaver, abc.ABC):
    SUB_FOLDER = Path("objects/meta_model")
    
    def set(self, model: OxariMetaModel) -> "MetaModelSaver":
        self._store = model
        return self
    
    @property
    def name(self):
        return f"meta_model_{self._name}_{self._time}"


class DataSaver(PartialSaver, abc.ABC):
    SUB_FOLDER = Path("objects/estimates")

    def set(self, dataset: OxariDataManager) -> "DataSaver":
        self._store = dataset
        return self

    @property
    def name(self):
        return f"estimates_{self._name}_{self._time}"


class LARModelSaver(PartialSaver, abc.ABC):
    SUB_FOLDER = Path("objects/lar_model")

    def set(self, model: OxariMetaModel) -> "LARModelSaver":
        self._store = model
        return self

    @property
    def name(self):
        return f"lar_model_{self._name}_{self._time}"

class LocalMetaModelSaver(LocalDestination, MetaModelSaver):

    def _save(self, **kwargs) -> bool:
        
        return pkl.dump(self._store, io.open(self.destination_path / f"{self.name}.pkl", 'wb'))


class LocalLARModelSaver(LocalDestination, LARModelSaver):

    def _save(self, **kwargs) -> bool:
        return pkl.dump(self._store, io.open(self.destination_path / f"{self.name}.pkl", 'wb'))


class LocalDataSaver(LocalDestination, DataSaver):

    def _save(self, **kwargs) -> bool:
        # TODO: Save all data stored in the manager
        # TODO: On Exception do fallback
        csv_name_1 = self.destination_path / f"scope_imputed_{self._time}_{self._name}.csv"
        csv_name_2 = self.destination_path / f"lar_imputed_{self._time}_{self._name}.csv"
        scope_imputed = self._store.get_data_by_name(OxariDataManager.IMPUTED_SCOPES)
        lar_imputed = self._store.get_data_by_name(OxariDataManager.IMPUTED_LARS)
        scope_imputed.to_csv(csv_name_1, index=False)
        lar_imputed.to_csv(csv_name_2, index=False)

class S3MetaModelSaver(S3Destination, MetaModelSaver):

    def _save(self, **kwargs):
        pkl_stream = pkl.dumps(self._store)
        self.client.put_object(Body=pkl_stream, Bucket='remote', Key=str(self.SUB_FOLDER / f"{self.name}.pkl"))

class S3LARModelSaver(S3Destination, LARModelSaver):

    def _save(self, **kwargs):
        pkl_stream = pkl.dumps(self._store)
        self.client.put_object(Body=pkl_stream, Bucket='remote', Key=str(self.SUB_FOLDER / f"{self.name}.pkl"))

class S3DataSaver(S3Destination, DataSaver):
    def _save(self, **kwargs) -> bool:
        # TODO: Save all data stored in the manager
        csv_name_1 = f"scope_imputed_{self.name}"
        csv_name_2 = f"lar_imputed_{self.name}"
        scope_imputed = self._store.get_data_by_name(OxariDataManager.IMPUTED_SCOPES)
        lar_imputed = self._store.get_data_by_name(OxariDataManager.IMPUTED_LARS)
        self.client.put_object(Body=scope_imputed.to_csv(index=False), Bucket='remote', Key=str(self.SUB_FOLDER / f"{csv_name_1}.csv"))
        self.client.put_object(Body=lar_imputed.to_csv(index=False), Bucket='remote', Key=str(self.SUB_FOLDER / f"{csv_name_2}.csv"))
        

class OxariSavingManager():
    """
    Saves the files into the appropriate location
    """

    def __init__(
        self,
        meta_model: MetaModelSaver = None,
        dataset: DataSaver = None,
        lar_model: LARModelSaver = None,
        other_savers: Dict[str, PartialSaver] = None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.saver_meta_model = meta_model
        self.saver_dataset = dataset
        self.saver_lar_model = lar_model
        # self.other_savers = categorical_loader
        self.verbose = verbose
        self._register_all_modules_to_pickle()

    def run(self, **kwargs) -> "OxariDataManager":
        self.saver_meta_model.save(**kwargs)
        self.saver_lar_model.save(**kwargs)
        self.saver_dataset.save(**kwargs)

    def _register_all_modules_to_pickle(self):
        # https://oegedijk.github.io/blog/pickle/dill/python/2020/11/10/serializing-dill-references.html
        # https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs
        for md in MODULES_TO_PICKLE:
            pkl.register_pickle_by_value(md)
