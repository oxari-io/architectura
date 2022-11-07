from base import pipeline
from base.dataset_loader import OxariDataLoader
from base import OxariScopeEstimator
from base import OxariPostprocessor
from base import OxariPreprocessor
from base.common import OxariImputer, OxariFeatureSelector
from dataset_loader.csv_loader import CSVDataLoader
from imputers.baseline import DummyImputer
from feature_selectors.baseline import DummyFeatureSelector
from preprocessors.baseline import BaselinePreprocessor
from scope_estimators.baseline import DefaultScopeEstimator


class DefaultPipeline(pipeline.OxariPipeline):
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
        self.dataset = dataset or CSVDataLoader()  ### None os CSVLoader OK
        self.preprocessor = preprocessor or BaselinePreprocessor()
        self.preprocessor = self.preprocessor.set_imputer(imputer or DummyImputer())
        self.preprocessor = self.preprocessor.set_feature_selector(feature_selector or DummyFeatureSelector())
        self.scope_estimator = scope_estimator or DefaultScopeEstimator()
        # postprocessor = DummyPostprocessor()
        super().__init__(dataset=self.dataset, preprocessor=self.preprocessor, scope_estimator=self.scope_estimator, postprocessor=None, database_deployer=None)

    def run_pipeline(self, **kwargs):
        # kwargs.pop("")
        self.dataset = self.dataset.run()
        df_processed = self.preprocessor.fit_transform(self.dataset._df_original)
        self.dataset = self.dataset.set_preprocessed_data(df_processed)
        # self.dataset = self.dataset.set_filled_data(df_filled)

        # TODO: Implement the train test split params as method parameters of run_pipeline
        # TODO: Run pipeline needs a parameter for which scope to run on
        X_train, y_train, X_rem, y_rem, X_test, y_test, X_val, y_val = self.dataset.train_test_val_split(
            scope=1,
            split_size_test=0.2,
            split_size_val=0.2,
            list_of_skipped_columns=["scope_1", "scope_2", "scope_3", "isin", "year"],
        )

        best_parameters, info = self.scope_estimator.optimize(X_train, y_train, X_val, y_val)

        self.scope_estimator = self.scope_estimator.fit(X_rem, y_rem, **best_parameters)
        y_pred = self.scope_estimator.predict(X_test)
        self.scope_estimator._evaluator.evaluate(y_pred, y_test)

        # self.postprocessor.fit()
        return self