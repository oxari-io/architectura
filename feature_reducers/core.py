from typing import Union
from base import OxariFeatureReducer
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.manifold import Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.base import BaseEstimator
from sklearn.decomposition import FactorAnalysis, LatentDirichletAllocation

# from factor_analyzer import FactorAnalyzer
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
class SKlearnFeatureReducerWrapperMixin():
    # def __init__(self,**kwargs):
    #     super().__init__(**kwargs)
    #     self._dimensionality_reducer:BaseEstimator = None
         
    def get_params(self, deep=False):
        return {**self._dimensionality_reducer.get_params(deep)}

# https://datascience.stackexchange.com/questions/29572/is-it-possible-to-do-feature-selection-for-unsupervised-machine-learning-problem
class DummyFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector does not select any feature. Use this if no feature selection is used."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        

    def fit(self, X, y=None, **kwargs) -> "OxariFeatureReducer":
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return X


class PCAFeatureSelector(OxariFeatureReducer, SKlearnFeatureReducerWrapperMixin):
    """ This Feature Selector uses PCA to reduce the dimensionality of the features first"""
    def __init__(self, n_components=5, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = PCA(n_components=n_components)

    def fit(self, X, y=None, **kwargs) -> "PCAFeatureSelector":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(X_new[self._features]), index=X_new.index)
        new_X_reduced = self.merge(X_new, reduced_features, self._features)
        return new_X_reduced

#The SKlearnFeatureReducerWrapperMixin was not an argument in the original iteration of this
class ModifiedLocallyLinearEmbedding(OxariFeatureReducer, SKlearnFeatureReducerWrapperMixin):
    """This Feature Selector results in a lower-dimensional projection of the data 
    which preserves distances within local neighborhoods. It additionally uses multiple 
    weight vectors in each neighborhood to solve the LLE regularisation problem"""
    def __init__(self, n_neighbors=5, n_components=5, method="modified", **kwargs):  #are kwargs the parameters of this estimator?
        super().__init__(**kwargs)
        self._dimensionality_reducer = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method)

    def fit(self, X, y=None, **kwargs) -> "OxariFeatureReducer":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        new_X_reduced = self.merge(new_X, reduced_features, self._features)
        return new_X_reduced

#The SKlearnFeatureReducerWrapperMixin was not an argument in the original iteration of this
class SpectralEmbedding(OxariFeatureReducer, SKlearnFeatureReducerWrapperMixin):
    """This Feature Selector finds a low dimensional representation of the data using 
    a spectral decomposition of the graph Laplacian"""
    def __init__(self, n_components=5, **kwargs):
        super().__init__(**kwargs)        
        self._dimensionality_reducer = SpectralEmbedding(n_components=n_components)

    def fit(self, X, y=None, **kwargs) -> "OxariFeatureReducer":
        # self._features = list(kwargs.get('features'))
        # self._dimensionality_reducer.fit(X[self._features], y)
        # self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        print("this is called")
        return self

    def transform(self, X, y=None, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        # self._dimensionality_reducer.fit(X[self._features], y)
        # new_X = X.copy()
        # reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        # new_X_reduced = self.merge(new_X, reduced_features, self._features)
        # return new_X_reduced
        print("this is also called")
        return self

    def fit_transform(self, X, y=None, **kwargs) -> Union[np.ndarray, pd.DataFrame]:  
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]

        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.fit_transform(new_X[self._features]), index=new_X.index)
        new_X_reduced = self.merge(new_X, reduced_features, self._features)
        print("this went through")
        # fit_transform will generate a new embedding space instead of projecting new points 
        # into the same embedding space used for the reference data. 
        # Plus the doc-page indicates that the output is an ndarray, not an array-like object (tho that's not an issue???)
        # new_X_reduced = new_X_reduced.reshape(-1)
        return new_X_reduced

class Factor_Analysis(OxariFeatureReducer):
    """This Feature Selector creates factors from the observed variables to represent the common variance 
    i.e. variance due to correlation among the observed variables."""
    # Number of components can (and maybe should) change
    # What's the effect of the rotation parameter? What if it's None?
    def __init__(self, n_components=5, rotation="varimax", **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = FactorAnalysis(n_components=n_components, rotation=rotation)

    def fit(self, X, y=None, **kwargs) -> "OxariFeatureReducer":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)        
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]      
        return self

    def transform(self, X:pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.fit_transform(new_X[self._features]), index=new_X.index)
        reduced_features.columns = [f"pc_{i}" for i in reduced_features.columns] 
        new_X = new_X.drop(columns=self._features)
        new_X = new_X.merge(reduced_features, left_index=True, right_index=True)
        return new_X

    
class Latent_Dirichlet_Allocation(OxariFeatureReducer):
    """This Feature Selector is a statistical technique that can extract underlying themes/topics 
    from a corpus."""
    # N_COMPONENTS DEFAULT IS 10
    # If the data size is large, the "ONLINE" update will be much faster than the "BATCH" update
    def __init__(self, n_components=5, learning_method="batch", **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = LatentDirichletAllocation(n_components=n_components, learning_method=learning_method)

    def fit(self, X, y=None, **kwargs) -> "OxariFeatureReducer":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)        
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X:pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        reduced_features.columns = [f"pc_{i}" for i in reduced_features.columns] 
        new_X = new_X.drop(columns=self._features)
        new_X = new_X.merge(reduced_features, left_index=True, right_index=True)
        return new_X





class IsomapFeatureSelector(OxariFeatureReducer, SKlearnFeatureReducerWrapperMixin):
    """ This Feature Selector uses Isomap manifold learning to reduce the dimensionality of the features"""
    def __init__(self, n_components=10, **kwargs):
        #TODO think about arguments of isomap
        super().__init__(**kwargs)        
        self._dimensionality_reducer = Isomap(n_components=n_components)

    # "Compute the embedding vectors for data X."
    def fit(self, X, y=None, **kwargs) -> "IsomapFeatureSelector":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X, y=None, **kwargs):
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.fit_transform(new_X[self._features]), index=new_X.index)
        new_X_reduced = self.merge(new_X, reduced_features, self._features)
        return new_X_reduced


class MDSSelector(OxariFeatureReducer, SKlearnFeatureReducerWrapperMixin):
    """ This Feature Selector uses Multidimensional Scaling
    
    You can find an explanation here: https://www.statisticshowto.com/multidimensional-scaling/ 
    """
    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)        
        self._dimensionality_reducer = MDS(n_components=n_components)

    "Compute the embedding vectors for data X."

    def fit(self, X, y=None, **kwargs) -> "MDSSelector":
        self._features = list(kwargs.get('features'))
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X, y=None, **kwargs):
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.fit_transform(new_X[self._features]), index=new_X.index)
        new_X_reduced = self.merge(new_X, reduced_features, self._features)
        return new_X_reduced


class DropFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector selects features according to a list of predefined features. 
    This is useful if a supervised feature elimination algorithm was used. 
    In other words, if the feature elimination algorithm cannot run during preprocessing.
    """
    def __init__(self, features=[], **kwargs):
        super().__init__(**kwargs)        
        self._features = features

    def fit(self, X, y=None, **kwargs) -> "DropFeatureReducer":
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.drop(columns=self._features)
        return new_X


class FeatureAgglomeration(OxariFeatureReducer):
    def __init__(self, features=[], **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = cluster.FeatureAgglomeration(n_clusters=17)

    def fit(self, X, y=None, **kwargs) -> "FeatureAgglomeration":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        # self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        # new_X_reduced = self.merge(new_X, reduced_features, self._features)
        # return new_X_reduced
        return reduced_features

class GaussRandProjection(OxariFeatureReducer):
    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = GaussianRandomProjection(n_components=n_components)

    def fit(self, X, y=None, **kwargs) -> "GaussRandProjection":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        # self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        # new_X_reduced = self.merge(new_X, reduced_features, self._features)
        # return new_X_reduced
        return reduced_features

class SparseRandProjection(OxariFeatureReducer):
    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = SparseRandomProjection(n_components=n_components)

    def fit(self, X, y=None, **kwargs) -> "SparseRandProjection":
        self._features = list(kwargs.get('features'))
        self._dimensionality_reducer.fit(X[self._features], y)
        # self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        # new_X_reduced = self.merge(new_X, reduced_features, self._features)
        # return new_X_reduced
        return reduced_features