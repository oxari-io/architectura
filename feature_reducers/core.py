from typing import Union
from base import OxariFeatureReducer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS

# https://datascience.stackexchange.com/questions/29572/is-it-possible-to-do-feature-selection-for-unsupervised-machine-learning-problem
class DummyFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector does not select any feature. Use this if no feature selection is used."""
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **kwargs) -> "OxariFeatureReducer":
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return X


class PCAFeatureSelector(OxariFeatureReducer):
    """ This Feature Selector uses PCA to reduce the dimensionality of the features first"""
    def __init__(self, n_components=5, **kwargs):
        self._dimensionality_reducer = PCA(n_components=n_components)

    def fit(self, X, y=None, **kwargs) -> "PCAFeatureSelector":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        return self

    def transform(self, X:pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        reduced_features.columns = [f"pc_{i}" for i in reduced_features.columns] 
        new_X = new_X.drop(columns=self._features)
        new_X = new_X.merge(reduced_features, left_index=True, right_index=True)
        return new_X

class IsomapFeatureSelector(OxariFeatureReducer):
    """ This Feature Selector uses Isomap manifold learning to reduce the dimensionality of the features"""
    def __init__(self, n_components=5, **kwargs):
        #TODO think about arguments of isomap
        self._dimensionality_reducer = Isomap(n_components=n_components)
    
    # "Compute the embedding vectors for data X."
    def fit(self, X, y=None, **kwargs) -> "IsomapFeatureSelector":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        return self

    def transform(self, X:pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        # TODO see if returning this directly makes sense
        return reduced_features

class MDSSelector(OxariFeatureReducer):
    """ This Feature Selector uses """
    def __init__(self, n_components=5, **kwargs):
        self._dimensionality_reducer = MDS(n_components=n_components)
    
    "Compute the embedding vectors for data X."
    def fit(self, X, y=None, **kwargs) -> "MDSSelector":
        pass

    def transform(self, X, y=None):
        pass

    def fit_transform(self, X:pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        self._features = list(kwargs.get('features'))
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.fit_transform(new_X[self._features]), index=new_X.index)
        # TODO see if returning this directly makes sense
        return reduced_features



class DropFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector selects features according to a list of predefined features. 
    This is useful if a supervised feature elimination algorithm was used. 
    In other words, if the feature elimination algorithm cannot run during preprocessing.
    """
    def __init__(self, features=[], **kwargs):
        self._features = features

    def fit(self, X, y=None, **kwargs) -> "DropFeatureReducer":
        return self

    def transform(self, X:pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.drop(columns = self._features)
        return new_X