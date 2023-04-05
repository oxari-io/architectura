from typing import Union
from typing_extensions import Self
from base.oxari_types import ArrayLike
import category_encoders as ce
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep

from base import OxariPreprocessor
from base.helper import DummyTargetScaler, OxariFeatureTransformerWrapper


class DummyPreprocessor(OxariPreprocessor):

    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        super().__init__(**kwargs)
        self.cat_transformer = cat_transformer or ce.OrdinalEncoder()
        self.fin_transformer = fin_transformer or DummyTargetScaler()

    def fit(self, X: pd.DataFrame, y, **kwargs) -> Self:
        data = X
        self.logger.info(f'number of original features: {len(data.columns)}')
        self.scope_columns = ["scope_1", "scope_2", "scope_3"]
        self.financial_columns = X.columns[X.columns.str.startswith('ft_num')]
        self.categorical_columns = X.columns[X.columns.str.startswith('ft_cat')]
        # # log scaling the scopes
        # self.scope_transformer = self.scope_transformer.fit(data[self.scope_columns])
        # transform numerical
        self.fin_transformer = self.fin_transformer.fit(data[self.financial_columns])
        # encode categorical
        self.cat_transformer = self.cat_transformer.fit(X=data[self.categorical_columns], y=(data[self.scope_columns][0]))
        return data

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        data = X
        # transform numerical
        data[self.financial_columns] = self.fin_transformer.transform(data[self.financial_columns])
        # encode categorical
        data[self.categorical_columns] = self.cat_transformer.transform(X=data[self.categorical_columns], y=(data[self.scope_columns[0]]))
        self.logger.info(f'number of features after preprocessing: {len(data.columns)}')
        return data


class BaselinePreprocessor(OxariPreprocessor):

    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        super().__init__(**kwargs)
        self.fin_transformer = fin_transformer or prep.RobustScaler()
        self.cat_transformer = cat_transformer or ce.TargetEncoder()
        # self.scope_transformer = scope_transformer or LogarithmScaler()
        # self.scope_columns = NumMapping.get_targets()
        # self.financial_columns = FinancialLoader.columns
        # self.categorical_columns = CategoricalLoader.columns

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "BaselinePreprocessor":
        data = X.copy()
        self.original_features = data.columns
        self.scope_columns = data.columns[data.columns.str.startswith('tg_num')]
        self.financial_columns = data.columns[data.columns.str.startswith('ft_num')]
        self.categorical_columns = data.columns[data.columns.str.startswith('ft_cat')]

        # transform numerical
        self.fin_transformer = self.fin_transformer.fit(data.loc[:, self.financial_columns])
        # encode categorical
        self.cat_transformer = self.cat_transformer.fit(data.loc[:, self.categorical_columns], y=np.array(y))
        # fill missing values
        self.imputer = self.imputer.fit(data.loc[:, self.financial_columns])
        self.logger.info(f'Preprocessed {len(self.original_features)} features to {len(self.financial_columns) + len(self.cat_transformer.cols)}')
        self.logger.info(f'{len(self.financial_columns)} numerical, {len(self.categorical_columns)} categorical columns')
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        X_result = X.copy()
        X_new = X.filter(regex='ft_', axis=1)
        # impute all the missing columns
        X_new.loc[:, self.financial_columns] = self.imputer.transform(X_new[self.financial_columns].astype(float))
        # transform numerical
        X_new.loc[:, self.financial_columns] = self.fin_transformer.transform(X_new[self.financial_columns])
        # encode categorical
        X_new.loc[:, self.categorical_columns] = self.cat_transformer.transform(X_new[self.categorical_columns])
        X_result[X_new.columns] = X_new[X_new.columns].values
        return X_result

    def get_config(self, deep=True):
        return {
            "fin_transformer": self.fin_transformer.__class__.__name__,
            "cat_transformer": self.cat_transformer.__class__.__name__,
            "imputer": self.imputer.__class__.__name__,
            **self.fin_transformer.get_params(deep),
            **self.cat_transformer.get_params(deep),
            **self.imputer.get_config(deep),
            **super().get_config(deep)
        }


class ImprovedBaselinePreprocessor(BaselinePreprocessor):

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        X_new = X.copy()
        self.scope_columns = X.columns[X.columns.str.startswith('tg_numc')]
        self.financial_columns = X.columns[X.columns.str.startswith('ft_num')]
        self.categorical_columns = X.columns[X.columns.str.startswith('ft_cat')]
        self.cat_transformer = self.cat_transformer.fit(X=X_new[self.categorical_columns], y=y)
        # TODO: Create an ABC for feature transformers. To allow adding full data and maintain a memory of which feature was transformed.
        # Similar to feature reducers.
        # Also allows to transform to a dataframe.
        X_new[self.categorical_columns] = self.cat_transformer.transform(X_new[self.categorical_columns])
        self.imputer = self.imputer.fit(X_new[self.financial_columns])
        self.fin_transformer = self.fin_transformer.fit(X_new)
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
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
        self.overall_scaler = OxariFeatureTransformerWrapper(transformer=prep.StandardScaler()) 

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # NOTE: Using fit_transform here leads to recursion.
        super().fit(X, y, **kwargs)
        X_new = super().transform(X, **kwargs)
        self.overall_scaler.fit(X_new)
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        X_new = super().transform(X, **kwargs)
        X_new = self.overall_scaler.transform(X_new)
        return X_new


class NormalizedIIDPreprocessor(IIDPreprocessor):
    """
    This preprocessor works well with the bayesian regressor. And probably neural networks.
    """

    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        super().__init__(fin_transformer, cat_transformer, **kwargs)
        self.overall_scaler_2 = OxariFeatureTransformerWrapper(transformer=prep.MinMaxScaler())

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # NOTE: Using fit_transform here leads to recursion.
        super().fit(X, y, **kwargs)
        X_new = super().transform(X, **kwargs)
        self.overall_scaler_2.fit(X_new)
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        X_new = super().transform(X, **kwargs)
        X_new = pd.DataFrame(self.overall_scaler_2.transform(X_new, **kwargs), index=X_new.index, columns=X_new.columns)
        return X_new