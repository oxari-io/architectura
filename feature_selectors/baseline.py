from typing import Union
from base.common import OxariFeatureSelector
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class DummyFeatureSelector(OxariFeatureSelector):
    """ This Feature Selector does not select any feature. Use this if no feature selection is used."""
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **kwargs) -> "OxariFeatureSelector":
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return X


class PCAFeatureSelector(OxariFeatureSelector):
    """ This Feature Selector uses PCA to reduce the dimensionality of the features first"""
    def __init__(self, n_components=5, **kwargs):
        self._dimensionality_reducer = PCA(n_components=n_components)

    def fit(self, X, y=None, **kwargs) -> "PCAFeatureSelector":
        self._dimensionality_reducer.fit(X, y)
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return pd.DataFrame(self._dimensionality_reducer.transform(X))


class PresetFeatureSelector(OxariFeatureSelector):
    """ This Feature Selector selects features according to a list of predefined features. 
    This is useful if a supervised feature elimination algorithm was used. 
    In other words, if the feature elimination algorithm cannot run during preprocessing.
    """
    def __init__(self, features=[], **kwargs):
        self.features = features

    def fit(self, X, y=None, **kwargs) -> "PresetFeatureSelector":
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X[self.features]
        return new_X