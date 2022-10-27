from base import pipeline
from base.dataset_loader import OxariDataLoader
from base import OxariScopeEstimator
from base import OxariPostprocessor
from base import OxariPreprocessor
from base.common import OxariImputer
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
        self.dataset = dataset or CSVDataLoader() ### None os CSVLoader OK
        self.preprocessor = preprocessor or BaselinePreprocessor(imputer=DummyImputer())
        self.imputer = imputer or OxariImputer()
        self.scope_estimator = DefaultScopeEstimator()
        # postprocessor = DummyPostprocessor()
        super().__init__(dataset, preprocessor, imputer, scope_estimator, postprocessor, database_deployer)

    def run_pipeline(self, **kwargs):
        # kwargs.pop("")
        df_processed = self.preprocessor.fit_transform(self.dataset._df_original)
        self.dataset = self.dataset.set_preprocessed_data(df_processed)
        df_filled = self.imputer.fit_transform(dataset._df_filled)
        self.dataset = self.dataset.set_filled_data(df_filled)
        
        X_train, y_train, X_train_full, y_train_full, X_test, y_test, X_val, y_val = self.dataset.train_test_val_split(scope=1)
        
        best_parameters, info =  self.scope_estimator._optimizer.optimize(X_train, y_train, X_val, y_val)
        
        self.scope_estimator = self.scope_estimator.fit(X_train_full, y_train_full, **best_parameters)
        y_pred = self.scope_estimator.predict(X_test)
        self.scope_estimator._evaluator.evaluate(y_pred, y_test)
        
        # self.postprocessor.fit()
        return self