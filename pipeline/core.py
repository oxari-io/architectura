from base.dataset_loader import OxariDataManager
from base import OxariScopeEstimator, OxariFeatureReducer, OxariPostprocessor, OxariPreprocessor, OxariPipeline, OxariConfidenceEstimator
from base import OxariImputer, JacknifeConfidenceEstimator

from dataset_loader.csv_loader import CSVDataManager
from imputers import BaselineImputer
from feature_reducers import DummyFeatureReducer
from preprocessors import BaselinePreprocessor
from scope_estimators import DummyEstimator
import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.model_selection import train_test_split

class DefaultPipeline(OxariPipeline):
    def __init__(
        self,
        # dataset: OxariDataLoader = None,
        preprocessor: OxariPreprocessor = None,
        feature_selector: OxariFeatureReducer = None,
        imputer: OxariImputer = None,
        scope_estimator: OxariScopeEstimator = None,
        ci_estimator: OxariConfidenceEstimator = None,
        **kwargs,
    ):
        super().__init__(
            # dataset=dataset or CSVDataLoader(),
            preprocessor=(preprocessor or BaselinePreprocessor()).set_imputer(imputer or BaselineImputer()),
            feature_selector=feature_selector or DummyFeatureReducer(),
            scope_estimator=scope_estimator or DummyEstimator(),
            ci_estimator = ci_estimator,
            **kwargs,
        )

    def optimise(self, X, y, **kwargs) -> OxariPipeline:
        # kwargs.pop("")
        # df_original = dataset.get_data_by_name('original')
        df_processed: pd.DataFrame = self.preprocessor.fit_transform(X,y)
        # dataset.add_data(f"scope_{self.scope}_processed", df_processed, "Dataset after preprocessing.")
        df_reduced: pd.DataFrame = self.feature_selector.fit_transform(df_processed, features=df_processed.columns)
        # dataset.add_data(f"scope_{self.scope}_reduced", df_reduced, "Dataset after feature selection.")
        X, y = df_reduced, y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)        
        self.params, self.info = self.estimator.optimize(X_train, y_train, X_test, y_test)
        return self

    def evaluate(self, X_train, y_train, X_val, y_val) -> OxariPipeline:
        X_val = self._preprocess(X_val)
        y_pred = self.estimator.predict(X_val)
        self._evaluation_results = self.estimator.evaluate(y_val, y_pred, X_test=X_val)
        return self


    @property
    def evaluation_results(self):
        # return {**super().evaluation_results}
        return {
            "imputer": self.preprocessor.imputer.name,
            "preprocessor": self.preprocessor.name,
            "feature_selector": self.feature_selector.name,
            "scope_estimator": self.estimator.name,
            
            **super().evaluation_results,
            "optimal_params":self.params,
        }

    # TODO: Should be get_config
    def get_config(self, deep=True):
        return {
            "preprocessor": self.preprocessor.get_config(deep),
            "feature_selector": self.feature_selector.get_config(deep),
            "scope_estimator": self.estimator.get_config(deep),
            **super().get_config(deep),
        }




class CVPipeline(DefaultPipeline):
    # TODO: Implement a version of the default pipeline which makes sure that a proper crossvalidation is run to evaluate the models. Might need to be handled on the Evaluator level.

    def evaluate(self, X_train, y_train, X_test, y_test, **kwargs):
        # TODO: This is just a start and not completed. It should use smape and R2. Also a postprocessor is probably more appropriate
        tmp_estimator = self.estimator.clone()
        X_train = self._preprocess(X_train, **kwargs)        
        X_test = self._preprocess(X_test, **kwargs)        
        scores = ms.cross_val_score(estimator=tmp_estimator, X=X_train, y=y_train, cv=10, scoring="neg_mean_absolute_percentage_error", error_score='raise')
        y_pred = self.estimator.predict(X_test)
        self._evaluation_results = {"crossval": {"mape": np.mean(scores)}, **self.estimator.evaluate(y_test, y_pred, X_test=X_test)}
        return self
