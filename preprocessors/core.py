from typing import Union
from typing_extensions import Self
from base.common import OxariFeatureTransformer
from base.oxari_types import ArrayLike
import category_encoders as ce
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep

from base import OxariPreprocessor
from base.helper import DummyTargetScaler, OxariFeatureTransformerWrapper
from preprocessors.helper.custom_cat_normalizers import CountryCodeCatColumnNormalizer, LinkTransformerCatColumnNormalizer, OxariCategoricalNormalizer, SectorNameCatColumnNormalizer, IndustryNameCatColumnNormalizer


class DummyPreprocessor(OxariPreprocessor):

    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        super().__init__(**kwargs)
        self.cat_transformer = cat_transformer or ce.OrdinalEncoder()
        self.fin_transformer = fin_transformer or DummyTargetScaler()

    def fit(self, X: pd.DataFrame, y, **kwargs) -> Self:
        data = X
        self.logger.info(f'number of original features: {len(data.columns)}')
        self.scope_columns = X.columns[X.columns.str.startswith('tg_num')]
        self.financial_columns = X.columns[X.columns.str.startswith('ft_num')]
        self.categorical_columns = X.columns[X.columns.str.startswith('ft_cat')]
        # # log scaling the scopes
        # self.scope_transformer = self.scope_transformer.fit(data[self.scope_columns])
        # transform numerical
        self.fin_transformer = self.fin_transformer.fit(data[self.financial_columns])
        # encode categorical
        self.cat_transformer = self.cat_transformer.fit(X=data[self.categorical_columns], y=y)
        self.imputer = self.imputer.fit(data.loc[:, self.financial_columns])
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        data = X
        # transform numerical
        financial_data = data[self.financial_columns].astype(float)
        imputed_values = self.imputer.transform(financial_data)        
        data.loc[:, self.financial_columns] = imputed_values
        data[self.financial_columns] = self.fin_transformer.transform(data[self.financial_columns])


        # encode categorical
        data[self.categorical_columns] = self.cat_transformer.transform(X=data[self.categorical_columns])
        self.logger.info(f'number of features after preprocessing: {len(data.columns)}')
        return data


class BaselinePreprocessor(OxariPreprocessor):

    def __init__(self, fin_transformer=None, cat_transformer=None, cat_normalizer:OxariCategoricalNormalizer=None, **kwargs):
        super().__init__(**kwargs)
        self.fin_transformer = fin_transformer or prep.PowerTransformer()
        self.cat_transformer = cat_transformer or ce.TargetEncoder()
        self.cat_normalizer = cat_normalizer or OxariCategoricalNormalizer(
            col_transformers=[
                # NOTE: the linktransformer currently does not work with macOS
                LinkTransformerCatColumnNormalizer(),
                CountryCodeCatColumnNormalizer()
                # SectorNameCatColumnNormalizer(), 
                # IndustryNameCatColumnNormalizer(),
            ]
        )
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
        
        # Normalize categorical columns
        self.cat_normalizer = self.cat_normalizer.fit(data.loc[:, self.categorical_columns], y=np.array(y))
        data_new = self.cat_normalizer.transform(data.loc[:, self.categorical_columns], y=np.array(y))
        
        # encode categorical
        self.cat_transformer = self.cat_transformer.fit(data_new.loc[:, self.categorical_columns], y=np.array(y))

        # fill missing values
        self.imputer = self.imputer.fit(data)
        self.logger.info(f'Preprocessed {len(self.original_features)} features to {len(self.financial_columns) + len(self.cat_transformer.cols)}')
        self.logger.info(f'{len(self.financial_columns)} numerical, {len(self.categorical_columns)} categorical columns')
        return self


    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        X_result = X.copy()
        X_new = X.filter(regex='ft_', axis=1).copy()
        # impute all the missing columns
        # financial_data = X_new[self.financial_columns].astype(float)
        self.logger.info(f"Imputing data using {self.imputer.__class__}")
        X_imputed = self.imputer.transform(X_new)
        X_new = X_imputed.copy()
        # transform numerical
        self.logger.info(f"Transform numerical data using {self.fin_transformer.__class__}")
        financial_data = X_new[self.financial_columns].copy()
        transformed_values = self.fin_transformer.transform(financial_data)
        X_new.loc[:, self.financial_columns] = transformed_values
        # normalize categorical
        self.logger.info(f"Normalizing categorical data using {self.cat_normalizer.__class__}")
        categorical_data = X_new[self.categorical_columns].copy()
        normalized_cat_data = self.cat_normalizer.transform(categorical_data)
        
        # encode categorical
        self.logger.info(f"Encoding categorical data using {self.cat_transformer.__class__}")
        transformed_cat_data = self.cat_transformer.transform(normalized_cat_data)
        X_new.loc[:, self.categorical_columns] = transformed_cat_data

        # Set all the values
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

class FastIndustryNormalisationBaselinePreprocessor(BaselinePreprocessor):
    def __init__(self, fin_transformer=None, cat_transformer=None, **kwargs):
        cat_normalizer = OxariCategoricalNormalizer(
                    col_transformers=[SectorNameCatColumnNormalizer(),
                                      IndustryNameCatColumnNormalizer(),
                                      CountryCodeCatColumnNormalizer()])
        super().__init__(fin_transformer, cat_transformer, cat_normalizer, **kwargs)

class ImprovedBaselinePreprocessor(BaselinePreprocessor):

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        X_new = X.copy()
        self.scope_columns = X.columns[X.columns.str.startswith('tg_numc')]
        self.financial_columns = X.columns[X.columns.str.startswith('ft_num')]
        self.categorical_columns = X.columns[X.columns.str.startswith('ft_cat')]
        
        # Normalize categorical columns
        self.cat_normalizer = self.cat_normalizer.fit(X_new.loc[:, self.categorical_columns], y=np.array(y))
        X_cat_normalized = self.cat_normalizer.transform(X_new.loc[:, self.categorical_columns], y=np.array(y))
        
        self.cat_transformer = self.cat_transformer.fit(X=X_cat_normalized, y=y)
        # TODO: Create an ABC for feature transformers. To allow adding full data and maintain a memory of which feature was transformed.
        # Similar to feature reducers.
        # Also allows to transform to a dataframe.
        X_new[self.categorical_columns] = self.cat_transformer.transform(X_new[self.categorical_columns])
        self.imputer = self.imputer.fit(X_new[self.financial_columns])
        self.fin_transformer = self.fin_transformer.fit(X_new)
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        X_new = X.copy()
        self.logger.info(f"Imputing data using {self.imputer.__class__}")
        X_new[self.financial_columns] = self.imputer.transform(X_new[self.financial_columns].astype(float))
        self.logger.info(f"Normalizing categorical data using {self.cat_normalizer.__class__}")
        X_new[self.categorical_columns] = self.cat_normalizer.transform(X_new[self.categorical_columns])
        self.logger.info(f"Encoding categorical data using {self.cat_transformer.__class__}")
        X_new[self.categorical_columns] = self.cat_transformer.transform(X_new[self.categorical_columns])
        self.logger.info(f"Transform numerical data using {self.fin_transformer.__class__}")
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
        self.logger.info('Scaling every feature towards IID')
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
        self.logger.info('Normalizing every feature between 0 and 1')
        X_new = super().transform(X, **kwargs)
        X_new = pd.DataFrame(self.overall_scaler_2.transform(X_new, **kwargs), index=X_new.index, columns=X_new.columns)
        return X_new
    
    
class ImprovedIIDPreprocessor(ImprovedBaselinePreprocessor):
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


class ImprovedNormalizedIIDPreprocessor(ImprovedIIDPreprocessor):
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