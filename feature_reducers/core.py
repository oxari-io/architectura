from typing import Union

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import (PCA, FactorAnalysis, LatentDirichletAllocation)
from sklearn.manifold import (MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding)
from sklearn.random_projection import (GaussianRandomProjection, SparseRandomProjection)
from typing_extensions import Self
from base import OxariFeatureReducer
from base.oxari_types import ArrayLike
from sklearn.base import BaseEstimator
from logging import Logger

# TODO: Issue with unknown attributes can be solved using Generics[T]
class SKlearnFeatureReducerWrapperBase(OxariFeatureReducer):
    _dimensionality_reducer:BaseEstimator 

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.feature_names_in_ = list(X.filter(regex="^ft_", axis=1).columns)
        self.logger.info(f'Number of features before feature reduction: {len(self.feature_names_in_)}')
        self._dimensionality_reducer.fit(X[self.feature_names_in_], y)
        self.n_components_ = self._dimensionality_reducer._n_features_out
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        # Ensures correct order
        X_new = X[self.feature_names_in_]
        X_reduced = pd.DataFrame(self._dimensionality_reducer.transform(X_new), index=X_new.index)
        X_new_reduced = self.merge(X_new, X_reduced)
        self.logger.info(f'Number of features after feature reduction: {len(X_new_reduced.columns)}')
        return X_new_reduced

    def get_params(self, deep=False):
        return {**self._dimensionality_reducer.get_params(deep)}


# https://datascience.stackexchange.com/questions/29572/is-it-possible-to-do-feature-selection-for-unsupervised-machine-learning-problem
class DummyFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector does not select any feature. Use this if no feature selection is used."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.n_components_ = X.shape[1]
        self.logger.info(f'number of features before feature reduction: {len(X.columns)}')
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        self.logger.info(f'number of features after feature reduction: {len(X.columns)}')
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
        self.logger.info(f"Number of features before feature reduction: {len(list(kwargs.get('features')))}")
        self.logger.info(f"Number of features before feature reduction: {len(self._features)}")
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        new_X = X.drop(columns=self._features)
        self.logger.info(f'Number of features after feature reduction: {len(new_X.columns)}')
        return new_X

# TODO: Align names
class PCAFeatureSelector(SKlearnFeatureReducerWrapperBase):
    """ This Feature Selector uses PCA to reduce the dimensionality of the features first"""

    def __init__(self, n_components=5, **kwargs):
        super().__init__(**kwargs)
        self.n_components_ = n_components
        self._dimensionality_reducer:PCA = PCA(n_components=n_components)

    def fit(self, X, y=None, **kwargs) -> Self:
        self.feature_names_in_ = list(X.filter(regex="^ft_", axis=1).columns)
        self.logger.info(f'Number of features before feature reduction: {len(self.feature_names_in_)}')
        self._dimensionality_reducer.fit(X[self.feature_names_in_], y) 
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)



class FeatureAgglomeration(SKlearnFeatureReducerWrapperBase):

    def __init__(self, features=[], **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = cluster.FeatureAgglomeration(n_clusters=17)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


#The SKlearnFeatureReducerWrapperMixin was not an argument in the original iteration of this
class ModifiedLocallyLinearEmbeddingFeatureReducer(SKlearnFeatureReducerWrapperBase):
    """This Feature Selector results in a lower-dimensional projection of the data 
    which preserves distances within local neighborhoods. It additionally uses multiple 
    weight vectors in each neighborhood to solve the LLE regularisation problem"""

    def __init__(self, n_neighbors=5, n_components=5, method="modified", **kwargs):  #are kwargs the parameters of this estimator?
        super().__init__(**kwargs)
        self._dimensionality_reducer = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method)

    def fit(self, X, y=None, **kwargs) -> Self:
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class FactorAnalysisFeatureReducer(SKlearnFeatureReducerWrapperBase):
    """This Feature Selector creates factors from the observed variables to represent the common variance 
    i.e. variance due to correlation among the observed variables."""

    # Number of components can (and maybe should) change
    # What's the effect of the rotation parameter? What if it's None?
    def __init__(self, n_components=5, rotation="varimax", **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = FactorAnalysis(n_components=n_components, rotation=rotation)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class LDAFeatureReducer(SKlearnFeatureReducerWrapperBase):
    """This Feature Selector is a statistical technique that can extract underlying themes/topics 
    from a corpus."""

    # N_COMPONENTS DEFAULT IS 10
    # If the data size is large, the "ONLINE" update will be much faster than the "BATCH" update
    def __init__(self, n_components=5, learning_method="batch", **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = LatentDirichletAllocation(n_components=n_components, learning_method=learning_method)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class IsomapDimensionalityReduction(SKlearnFeatureReducerWrapperBase):
    """ This Feature Selector uses Isomap manifold learning to reduce the dimensionality of the features"""

    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = Isomap(n_components=n_components)

    # "Compute the embedding vectors for data X."
    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class GaussRandProjection(OxariFeatureReducer):

    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = GaussianRandomProjection(n_components=n_components)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class SparseRandProjection(OxariFeatureReducer):

    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = SparseRandomProjection(n_components=n_components)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        super().fit(X, y, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        return super().transform(X, **kwargs)


class MDSDimensionalitySelector(OxariFeatureReducer):
    """ This Feature Selector uses Multidimensional Scaling
    
    You can find an explanation here: https://www.statisticshowto.com/multidimensional-scaling/ 
    """

    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = MDS(n_components=n_components)

    "Compute the embedding vectors for data X."

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # self._features = list(kwargs.get('features'))
        # self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        # new_X = X.copy()
        # reduced_features = pd.DataFrame(self._dimensionality_reducer.fit_transform(new_X[self._features]), index=new_X.index)
        # new_X_reduced = self.merge(new_X, reduced_features, self._features)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        self._features = list(kwargs.get('features'))
        self.logger.info(f'Number of components before dimensionality reduction: {len(self._features)}')
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]

        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.fit_transform(new_X[self._features]), index=new_X.index)
        new_X_reduced = self.merge(new_X, reduced_features, self._features)
        self.logger.info(f'Number of components after dimensionality reduction: {len(new_X_reduced.columns)}')
        # fit_transform will generate a new embedding space instead of projecting new points
        # into the same embedding space used for the reference data.
        # Plus the doc-page indicates that the output is an ndarray, not an array-like object (tho that's not an issue???)
        # new_X_reduced = new_X_reduced.reshape(-1)
        return new_X_reduced


#The SKlearnFeatureReducerWrapperMixin was not an argument in the original iteration of this
class Spectral_Embedding(OxariFeatureReducer):
    """This Feature Selector finds a low dimensional representation of the data using 
    a spectral decomposition of the graph Laplacian"""

    def __init__(self, n_components=5, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = SpectralEmbedding(n_components=n_components)

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        return self

    def transform(self, X, y=None, **kwargs) -> ArrayLike:
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> ArrayLike:
        self._features = list(kwargs.get('features'))
        self.logger.info(f'Number of components before dimensionality reduction: {len(self._features)}')
        # self._dimensionality_reducer.fit(X[self._features], y)
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]

        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.fit_transform(new_X[self._features]), index=new_X.index)
        new_X_reduced = self.merge(new_X, reduced_features, self._features)
        self.logger.info(f'Number of components after dimensionality reduction: {len(new_X_reduced.columns)}')
        # fit_transform will generate a new embedding space instead of projecting new points
        # into the same embedding space used for the reference data.
        # Plus the doc-page indicates that the output is an ndarray, not an array-like object (tho that's not an issue???)
        # new_X_reduced = new_X_reduced.reshape(-1)
        return new_X_reduced