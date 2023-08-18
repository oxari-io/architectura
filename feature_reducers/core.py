from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import (PCA, FactorAnalysis, LatentDirichletAllocation)
from sklearn.manifold import (Isomap, LocallyLinearEmbedding)
from sklearn.random_projection import (GaussianRandomProjection, SparseRandomProjection)
from typing_extensions import Self
from base import OxariFeatureReducer
from base.oxari_types import ArrayLike
from sklearn.base import BaseEstimator
from logging import Logger

# TODO: Issue with unknown attributes can be solved using Generics[T]
# Let inherit from OxariFeatureReducer and specialise the others


# https://datascience.stackexchange.com/questions/29572/is-it-possible-to-do-feature-selection-for-unsupervised-machine-learning-problem
class DummyFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector does not select any feature. Use this if no feature selection is used."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.feature_names_in_ = list(X.filter(regex="^ft_", axis=1).columns)
        self.n_components_ = X.shape[1]
        self.logger.info(f'Reduces features from {self.n_components_} to {self.n_components_} columns. ')
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        return X


class DropFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector selects features according to a list of predefined features. 
    This is useful if a supervised feature elimination algorithm was used. 
    In other words, if the feature elimination algorithm cannot run during preprocessing.
    """

    def __init__(self, features=[], **kwargs):
        super().__init__(**kwargs)
        self._features = features

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.feature_names_in_ = list(X.filter(regex="^ft_", axis=1).columns)
        self.n_components_ = len(X.columns) - len(self._features)
        self.logger.info(f'Reduces features from {len(X[self.feature_names_in_].columns)} columns to {self.n_components_} columns.')
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        new_X = X.drop(columns=self._features)
        return new_X

class SelectionFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector selects features according to a list of predefined features. 
    This is useful if a supervised feature selection algorithm was used. 
    """

    def __init__(self, features=[], **kwargs):
        super().__init__(**kwargs)
        self._features = features

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.feature_names_in_ = list(X.filter(regex="^ft_", axis=1).columns)
        self.selected_features_ = set(X.columns).intersection(set(self._features))
        self.n_components_ = len(self.selected_features_)
        self.logger.info(f'Reduces features from {len(X[self.feature_names_in_].columns)} columns to {self.n_components_} columns.')
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        X_reduced = X[self.selected_features_]
        # TODO: add merge function with metadata
        return X_reduced


# TODO: Align names
class PCAFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector uses PCA to reduce the dimensionality of the features first"""

    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self.n_components_ = n_components
        self._dimensionality_reducer: PCA = PCA(n_components=n_components)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # take only the names of features that should be reduced
        self.feature_names_in_ = list(set(X.filter(regex="^ft_", axis=1).columns).difference(set(self.ignored_features_)))
        self._dimensionality_reducer.fit(X[self.feature_names_in_], y)
        self.logger.info(f'Reduces features from {len(X[self.feature_names_in_].columns)} columns to {self.n_components_} columns.')
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        # Ensures correct order
        X_new = X[self.feature_names_in_]
        X_reduced = pd.DataFrame(self._dimensionality_reducer.transform(X_new), index=X_new.index)
        X_new_reduced = self.merge(X_new, X_reduced)
        X_complete = self.merge_with_ignored_columns(X, X_new_reduced)
        return X_complete

    def get_params(self, deep=False):
        return {**self._dimensionality_reducer.get_params(deep)}

    def get_config(self, deep=True):
        return {'estimator': self._dimensionality_reducer.get_params(deep), **super().get_config(deep)}


class AgglomerateFeatureReducer(PCAFeatureReducer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer: FeatureAgglomeration = FeatureAgglomeration(n_clusters=17)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        super().fit(X, y, **kwargs)
        self.n_components_ = self._dimensionality_reducer._n_features_out
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


#The SKlearnFeatureReducerWrapperMixin was not an argument in the original iteration of this
class ModifiedLocallyLinearEmbeddingFeatureReducer(AgglomerateFeatureReducer):
    """This Feature Selector results in a lower-dimensional projection of the data 
    which preserves distances within local neighborhoods. It additionally uses multiple 
    weight vectors in each neighborhood to solve the LLE regularisation problem"""

    def __init__(self, n_components=10, n_neighbors=10,  method="modified", **kwargs):  #are kwargs the parameters of this estimator?
        super().__init__(**kwargs)
        self._dimensionality_reducer = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method, eigen_solver='dense')

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class FactorAnalysisFeatureReducer(AgglomerateFeatureReducer):
    """This Feature Selector creates factors from the observed variables to represent the common variance 
    i.e. variance due to correlation among the observed variables."""

    # Number of components can (and maybe should) change
    # What's the effect of the rotation parameter? What if it's None?
    def __init__(self, n_components=10, rotation="varimax", **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = FactorAnalysis(n_components=n_components, rotation=rotation)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class LDAFeatureReducer(AgglomerateFeatureReducer):
    """This Feature Selector is a statistical technique that can extract underlying themes/topics 
    from a corpus."""

    # N_COMPONENTS DEFAULT IS 10
    # If the data size is large, the "ONLINE" update will be much faster than the "BATCH" update
    def __init__(self, n_components=10, learning_method="batch", **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = LatentDirichletAllocation(n_components=n_components, learning_method=learning_method)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class IsomapDimensionalityFeatureReducer(AgglomerateFeatureReducer):
    """ This Feature Selector uses Isomap manifold learning to reduce the dimensionality of the features"""

    def __init__(self, n_components=10, n_neighbors=20, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = Isomap(n_components=n_components, n_neighbors=n_neighbors)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class GaussRandProjectionFeatureReducer(AgglomerateFeatureReducer):

    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = GaussianRandomProjection(n_components=n_components)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class SparseRandProjectionFeatureReducer(AgglomerateFeatureReducer):

    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = SparseRandomProjection(n_components=n_components)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)
