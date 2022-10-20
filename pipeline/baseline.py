from base import pipeline
from base.dataset_loader import OxariDataLoader
from base.estimator import OxariScopeEstimator
from base.imputer import OxariImputer
from base.postprocessor import OxariPostprocessor
from base.preprocessor import OxariPreprocessor
from dataset_loader.csv_loader import CSVDataLoader
from imputers.baseline import DummyImputer
from preprocessors.baseline import BaselinePreprocessor
from scope_estimators.baseline import DefaultScopeEstimator


class DefaultPipeline(pipeline.OxariPipeline):
    def __init__(
        self,
        dataset: OxariDataLoader = None,
        preprocessor: OxariPreprocessor = None,
        imputer: OxariImputer = None,
        scope_estimator: OxariScopeEstimator = None,
        postprocessor: OxariPostprocessor = None,
        database_deployer=None,
    ):
        dataset = dataset or CSVDataLoader()
        preprocessor = preprocessor or BaselinePreprocessor()
        imputer = imputer or DummyImputer()
        scope_estimator = DefaultScopeEstimator()
        # postprocessor = DummyPostprocessor()
        super().__init__(dataset, preprocessor, imputer, scope_estimator, postprocessor, database_deployer)

    def run_pipeline(self, **kwargs):
        dataset = self.dataset.set_preprocessed_data(self.preprocessor.fit_transform(dataset))
        dataset = self.dataset.set_filled_data(self.imputer.fit_transform(dataset.preprocessed_data))
        self.scope_estimator.fit(dataset.X, dataset.y)
        # self.postprocessor.fit()
        return self