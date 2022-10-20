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
from base.mappings import CatMapping, NumMapping


class DefaultPreprocessor(preprocessor.OxariPreprocessor):
    def __init__(self, scope_transformer=LogarithmScaler(), fin_transformer=RobustScaler(), cat_transformer=TargetEncoder(), **kwargs):
        super().__init__(**kwargs)
        self.fin_transformer = fin_transformer
        self.cat_transformer = cat_transformer
        self.scope_transformer = scope_transformer
        self.scope_columns = NumMapping.get_targets()
        self.financial_columns = NumMapping.get_features()
        self.categorical_columns = CatMapping.get_features()

    def run(self, **kwargs) -> "DefaultPreprocessor":
        return super().run(**kwargs)

    def fit(self, X:pd.DataFrame, y, **kwargs) -> "DefaultPreprocessor":
        data = X
        # log scaling the scopes
        self.scope_transformer = self.scope_transformer.fit(data[self.scope_columns])
        # transform numerical
        self.fin_transformer = self.fin_transformer.fit(data[self.financial_columns])
        # encode categorical
        self.cat_transformer = self.cat_transformer.fit(X=data[self.categorical_columns], y=(data[self.scope_columns][0]))
        return data

    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        data = X
        # log scaling the scopes -> NOTE: NOT NECESSARY DURING INFERENCE
        data[self.scope_columns] = self.scope_transformer.transform(data[self.scope_columns])
        # transform numerical
        data[self.financial_columns] = self.fin_transformer.transform(data[self.financial_columns])
        # encode categorical
        data[self.categorical_columns] = self.cat_transformer.transform(X=data[self.categorical_columns], y=(data[self.scope_columns[0]]))
        return data