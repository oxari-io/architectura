from base.dataset_loader import OxariDataManager
from base import  OxariPipeline
from base import DummyConfidenceEstimator, DummyScaler
from base.helper import LogarithmScaler

from dataset_loader.csv_loader import DefaultDataManager
from imputers import BaselineImputer
from feature_reducers import DummyFeatureReducer
from preprocessors import BaselinePreprocessor
from scope_estimators import DummyEstimator
import numpy as np
import pandas as pd
from sklearn import model_selection as ms

class DefaultPipeline(OxariPipeline):
    def __init__(
        self,
        # dataset: OxariDataLoader = None,
        **kwargs,
    ):
        super().__init__(
            # dataset=dataset or CSVDataLoader(),
            preprocessor=kwargs.pop('preprocessor', BaselinePreprocessor()).set_imputer(kwargs.pop('imputer', BaselineImputer())),
            feature_selector=kwargs.pop('feature_reducer', DummyFeatureReducer()),
            scope_estimator= kwargs.pop('scope_estimator', DummyEstimator()),
            ci_estimator = kwargs.pop('ci_estimator', DummyConfidenceEstimator()),
            scope_transformer = kwargs.pop('scope_transformer', DummyScaler()),
            **kwargs,
        )

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
            "cfg":self.get_config(),
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
        # y_test = self._transform_scope(y_test)
        # TODO: Needs rework so that all the estimators are actually using their params.
        scores = ms.cross_val_score(estimator=tmp_estimator, X=X_train, y=y_train, cv=10, scoring="neg_mean_absolute_percentage_error", error_score='raise')
        y_pred = self.estimator.predict(X_test)
        y_pred = self.scope_transformer.reverse_transform(y_pred)
        self._evaluation_results = {"crossval": {"mape": np.mean(scores)}, **self.estimator.evaluate(y_test, y_pred, X_test=X_test)}
        return self


def FSExperimentPipeline(DefaultPipeline):
    pass