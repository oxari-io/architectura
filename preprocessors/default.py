import abc
from typing import Any, Union
from base import preprocessor
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from category_encoders.target_encoder import TargetEncoder
import sklearn
from base.common import LogarithmScaler, OxariMixin, OxariTransformer
from base.dataset_loader import OxariDataLoader
import numpy as np
import pandas as pd


class DefaultPreprocessor(preprocessor.OxariPreprocessor):
    def __init__(self, scope_transformer=LogarithmScaler(), fin_transformer=RobustScaler(), cat_transformer=TargetEncoder(), **kwargs):
        super().__init__(**kwargs)
        self.fin_transformer = fin_transformer
        self.cat_transformer = cat_transformer
        self.scope_transformer = scope_transformer

    def run(self, **kwargs) -> "DefaultPreprocessor":
        return super().run(**kwargs)

    def fit(self, X: OxariDataLoader, y: Any, **kwargs) -> "DefaultPreprocessor":
        data = X.original_data
        # log scaling the scopes
        self.scope_transformer = self.scope_transformer.fit(data[X.scope_loader.columns])
        # transform numerical
        self.fin_transformer = self.fin_transformer.fit(data[X.financial_loader.columns])
        # encode categorical
        self.cat_transformer = self.cat_transformer.fit(X=data[X.categorical_loader.columns], y=(data[X.scope_loader.columns].sum(axis=1) / 3))
        return data

    def transform(self, X: OxariDataLoader, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        data = X.original_data
        # log scaling the scopes -> NOTE: NOT NECESSARY DURING INFERENCE
        data[X.scope_loader.columns] = self.scope_transformer.transform(data[X.scope_loader.columns])
        # transform numerical
        data[X.financial_loader.columns] = self.fin_transformer.transform(data[X.financial_loader.columns])
        # encode categorical
        data[X.categorical_loader.columns] = self.cat_transformer.transform(X=data[X.categorical_loader.columns], y=(data[X.scope_loader.columns].sum(axis=1) / 3))
        return data