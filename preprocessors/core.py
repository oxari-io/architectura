import abc
from typing import Any, Union
from base import OxariPreprocessor
import sklearn.preprocessing as prep
import category_encoders as ce
import sklearn
from base.common import OxariMixin, OxariTransformer, DummyScaler
from base.dataset_loader import OxariDataManager
import numpy as np
import pandas as pd
from base.mappings import CatMapping, NumMapping


class DummyPreprocessor(OxariPreprocessor):
    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        super().__init__(**kwargs)
        self.cat_transformer = cat_transformer or ce.OrdinalEncoder()
        self.fin_transformer = fin_transformer or DummyScaler()
        self.scope_columns = NumMapping.get_targets()
        self.financial_columns = NumMapping.get_features()
        self.categorical_columns = CatMapping.get_features()

    def run(self, **kwargs) -> "DummyPreprocessor":
        return super().run(**kwargs)

    def fit(self, X: pd.DataFrame, y, **kwargs) -> "DummyPreprocessor":
        data = X
        # # log scaling the scopes
        # self.scope_transformer = self.scope_transformer.fit(data[self.scope_columns])
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


class BaselinePreprocessor(OxariPreprocessor):
    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        super().__init__(**kwargs)
        self.fin_transformer = fin_transformer or prep.RobustScaler()
        self.cat_transformer = cat_transformer or ce.TargetEncoder()
        # self.scope_transformer = scope_transformer or LogarithmScaler()
        # self.scope_columns = NumMapping.get_targets()
        self.financial_columns = NumMapping.get_features()
        self.categorical_columns = CatMapping.get_features()

    def run(self, **kwargs) -> "BaselinePreprocessor":
        return super().run(**kwargs)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "BaselinePreprocessor":
        data = X

        # # log scaling the scopes
        # self.scope_transformer = self.scope_transformer.fit(data[self.scope_columns])
        # transform numerical
        self.fin_transformer = self.fin_transformer.fit(data[self.financial_columns])
        # encode categorical
        self.cat_transformer = self.cat_transformer.fit(X=data[self.categorical_columns], y=np.array(y))
        # fill missing values
        self.imputer = self.imputer.fit(data[self.financial_columns])
        # reduce dimensionality/feature count
        # self.feature_selector = self.feature_selector.fit(data.drop(columns=self.scope_columns + ["year", "isin"]))
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = X.copy()
        # impute all the missing columns
        X_new[self.financial_columns] = self.imputer.transform(X_new[self.financial_columns].astype(float))
        # log scaling the scopes -> NOTE: NOT NECESSARY DURING INFERENCE
        # data[self.scope_columns] = self.scope_transformer.transform(data[self.scope_columns])
        # transform numerical
        X_new[self.financial_columns] = self.fin_transformer.transform(X_new[self.financial_columns])
        # encode categorical
        X_new[self.categorical_columns] = self.cat_transformer.transform(X_new[self.categorical_columns])
        # reduce dimensionality/feature count
        # data = self.feature_selector.transform(data)
        return X_new

    def get_config(self, deep=True):
        return {
            **self.fin_transformer.get_params(deep),
            # **self.scope_transformer.get_params(deep),
            **self.cat_transformer.get_params(deep),
            **self.imputer.get_config(deep),
            **super().get_config(deep)
        }

class ImprovedBaselinePreprocessor(BaselinePreprocessor):

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "BaselinePreprocessor":
        X_new = X.copy()
        self.cat_transformer = self.cat_transformer.fit(X=X_new[self.categorical_columns], y=y)
        # TODO: Create an ABC for feature transformers. To allow adding full data and maintain a memory of which feature was transformed. 
        # Similar to feature reducers.
        # Also allows to transform to a dataframe.
        X_new[self.categorical_columns] = self.cat_transformer.transform(X_new[self.categorical_columns]) 
        self.imputer = self.imputer.fit(X_new[self.financial_columns])
        self.fin_transformer = self.fin_transformer.fit(X_new)
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = X.copy()
        X_new[self.financial_columns] = self.imputer.transform(X_new[self.financial_columns].astype(float))
        X_new[self.categorical_columns] = self.cat_transformer.transform(X_new[self.categorical_columns])
        X_new = pd.DataFrame(self.fin_transformer.transform(X_new), index=X_new.index, columns=X_new.columns)
        return X_new
    
class IIDPreprocessor(BaselinePreprocessor):
    """
    This preprocessor works well with the bayesian regressor. And probably neural networks.
    """
    
    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        super().__init__(fin_transformer, cat_transformer, **kwargs)
        self.overall_scaler = prep.StandardScaler()

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "BaselinePreprocessor":
        # NOTE: Using fit_transform here leads to recursion.
        super().fit(X, y, **kwargs)
        X_new = super().transform(X, **kwargs)
        self.overall_scaler.fit(X_new)
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = super().transform(X, **kwargs)
        X_new = pd.DataFrame(self.overall_scaler.transform(X_new), index=X_new.index, columns=X_new.columns)
        return X_new


class NormalizedIIDPreprocessor(IIDPreprocessor):
    """
    This preprocessor works well with the bayesian regressor. And probably neural networks.
    """
    
    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        super().__init__(fin_transformer, cat_transformer, **kwargs)
        self.overall_scaler_2 = prep.MinMaxScaler()

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "NormalizedIIDPreprocessor":
        # NOTE: Using fit_transform here leads to recursion.
        super().fit(X, y, **kwargs)
        X_new = super().transform(X, **kwargs)
        self.overall_scaler_2.fit(X_new)
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = super().transform(X, **kwargs)
        X_new = pd.DataFrame(self.overall_scaler_2.transform(X_new, **kwargs), index=X_new.index, columns=X_new.columns)
        return X_new