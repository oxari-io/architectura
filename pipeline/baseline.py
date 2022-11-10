from base.dataset_loader import OxariDataLoader
from base import OxariScopeEstimator, OxariFeatureSelector, OxariPostprocessor, OxariPreprocessor, OxariPipeline
from base.common import OxariImputer
from dataset_loader.csv_loader import CSVDataLoader
from imputers.baseline import DummyImputer
from feature_reducers.baseline import DummyFeatureSelector
from preprocessors.baseline import BaselinePreprocessor
from scope_estimators.baseline import DefaultScopeEstimator
import numpy as np
import pandas as pd


class DefaultPipeline(OxariPipeline):
    def __init__(
        self,
        # dataset: OxariDataLoader = None,
        preprocessor: OxariPreprocessor = None,
        feature_selector: OxariFeatureSelector = None,
        imputer: OxariImputer = None,
        scope_estimator: OxariScopeEstimator = None,
        postprocessor: OxariPostprocessor = None,
        database_deployer=None,
    ):
        super().__init__(
            # dataset=dataset or CSVDataLoader(),
            preprocessor=(preprocessor or BaselinePreprocessor()).set_imputer(imputer or DummyImputer()),
            feature_selector=feature_selector or DummyFeatureSelector(),
            scope_estimator=scope_estimator or DefaultScopeEstimator(),
            postprocessor=None,
            database_deployer=None,
        )

    def run_pipeline(self, dataset: OxariDataLoader, scope:int, **kwargs):
        # kwargs.pop("")
        list_of_skipped_columns = ["scope_1", "scope_2", "scope_3", "isin", "year"]
        df_original = dataset.get_data_by_name('original')
        df_processed: pd.DataFrame = self.preprocessor.fit_transform(df_original)
        dataset.add_data(f"scope_{scope}_processed", df_processed, "Dataset after preprocessing.")
        df_reduced: pd.DataFrame = self.feature_selector.fit_transform(df_processed, features=df_original.columns.difference(list_of_skipped_columns))
        dataset.add_data(f"scope_{scope}_reduced", df_reduced, "Dataset after feature selection.")
        X, y = df_reduced.drop(columns=list_of_skipped_columns), df_reduced[f"scope_{scope}"]

        # TODO: Implement the train test split params as method parameters of run_pipeline
        # TODO: Run pipeline needs a parameter for which scope to run on
        X_rem, y_rem, X_train, y_train, X_val, y_val, X_test, y_test = OxariDataLoader.train_test_val_split(
            X=X,
            y=y,
            split_size_test=0.2,
            split_size_val=0.2,
        )

        best_parameters, info = self.scope_estimator.optimize(X_train, y_train, X_val, y_val)
        # info.to_csv('optimization_results.csv')
        self.scope_estimator = self.scope_estimator.fit(X_rem, y_rem, **best_parameters)
        y_pred = self.scope_estimator.predict(X_test)
        self.evaluation_results = self.scope_estimator.evaluate(y_test, y_pred, X_test=X_test)
        # print(evaluation_results)
        return self

    def predict(self, X, **kwargs):
        return self.scope_estimator.predict(X, **kwargs)