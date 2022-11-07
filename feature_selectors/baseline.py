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
        self._features = list(X.columns)
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        new_X = new_X.drop(columns=self._features)
        new_X = new_X.merge(reduced_features)
        return new_X


class PresetFeatureSelector(OxariFeatureSelector):
    """ This Feature Selector selects features according to a list of predefined features. 
    This is useful if a supervised feature elimination algorithm was used. 
    In other words, if the feature elimination algorithm cannot run during preprocessing.
    """
    def __init__(self, features=[], **kwargs):
        self._features = features

    def fit(self, X, y=None, **kwargs) -> "PresetFeatureSelector":
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X[self._features]
        return new_X