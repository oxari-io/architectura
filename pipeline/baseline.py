from base.dataset_loader import OxariDataLoader
from base import OxariScopeEstimator, OxariFeatureSelector, OxariPostprocessor, OxariPreprocessor, OxariPipeline
from base.common import OxariImputer
from dataset_loader.csv_loader import CSVDataLoader
from imputers.baseline import DummyImputer
from feature_reducers.baseline import DummyFeatureSelector
from preprocessors.baseline import BaselinePreprocessor
from scope_estimators.baseline import DefaultScopeEstimator
import numpy as np

class DefaultPipeline(OxariPipeline):
    def __init__(
        self,
        dataset: OxariDataLoader = None,
        preprocessor: OxariPreprocessor = None,
        feature_selector: OxariFeatureSelector = None,
        imputer: OxariImputer = None,
        scope_estimator: OxariScopeEstimator = None,
        postprocessor: OxariPostprocessor = None,
        database_deployer=None,
    ):
        super().__init__(
            dataset=dataset or CSVDataLoader(),
            preprocessor=(preprocessor or BaselinePreprocessor()).set_imputer(imputer or DummyImputer()),
            feature_selector=feature_selector or DummyFeatureSelector(),
            scope_estimator=scope_estimator or DefaultScopeEstimator(),
            postprocessor=None,
            database_deployer=None,
        )

    def run_pipeline(self, **kwargs):
        # kwargs.pop("")
        list_of_skipped_columns = ["scope_1", "scope_2", "scope_3", "isin", "year"]
        self.dataset = self.dataset.run()
        df_processed = self.preprocessor.fit_transform(self.dataset.data)
        self.dataset = self.dataset.add_data("processed", df_processed, "Dataset after preprocessing.")
        df_reduced = self.feature_selector.fit_transform(self.dataset.data, features=set(self.dataset.data.columns)-set(list_of_skipped_columns))
        self.dataset = self.dataset.add_data("reduced", df_reduced, "Dataset after feature selection.")

        # TODO: Implement the train test split params as method parameters of run_pipeline
        # TODO: Run pipeline needs a parameter for which scope to run on
        X_train, y_train, X_rem, y_rem, X_test, y_test, X_val, y_val = self.dataset.train_test_val_split(
            scope=1,
            split_size_test=0.2,
            split_size_val=0.2,
            list_of_skipped_columns=list_of_skipped_columns,
        )

        best_parameters, info = self.scope_estimator.optimize(X_train, y_train, X_val, y_val)

        self.scope_estimator = self.scope_estimator.fit(X_rem, y_rem, **best_parameters)
        y_pred = self.scope_estimator.predict(X_test)
        self.scope_estimator.evaluate(y_test, y_pred, X_test=X_test)

        # self.postprocessor.fit()
        return self