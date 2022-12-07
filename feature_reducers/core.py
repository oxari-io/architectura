from typing import Union
from base import OxariFeatureReducer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.base import BaseEstimator
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


class SpectralEmbedding(OxariFeatureReducer, SKlearnFeatureReducerWrapperMixin):
    """This Feature Selector finds a low dimensional representation of the data using 
    a spectral decomposition of the graph Laplacian"""
    def __init__(self, n_components=5, **kwargs):
        super().__init__(**kwargs)        
        self._dimensionality_reducer = SpectralEmbedding(n_components=n_components)

    def fit(self, X, y=None, **kwargs) -> "OxariFeatureReducer":
        self._features = list(kwargs.get('features'))
        self.reduced_feature_columns = [f"ft_{i}" for i in range(self._dimensionality_reducer.n_components)]
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        self._dimensionality_reducer.fit(X[self._features], y)
        new_X = X.copy()
        reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
        new_X_reduced = self.merge(new_X, reduced_features, self._features)
        return new_X_reduced

    #DOING THIS BECAUSE THERE IS NO "TRANSFORM" METHOD; THERE IS A FIT METHOD BUT IF WE"RE USING
    #THE FIT_TRANSFORM METHOD THEN THE FIT METHOD MUST BE REDUNDANT
    # def fit_transform(self, X:pd.DataFrame, y=None, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    #     self._features = list(kwargs.get('features'))
    #     self._dimensionality_reducer.fit(X[self._features], y)
    #     new_X = X.copy()
    #     reduced_features = pd.DataFrame(self._dimensionality_reducer.transform(new_X[self._features]), index=new_X.index)
    #     reduced_features.columns = [f"pc_{i}" for i in reduced_features.columns]
    #     new_X = new_X.drop(columns=self._features)
    #     new_X = new_X.merge(reduced_features, left_index=True, right_index=True)
    #     return new_X


# class FactorAnalysis(OxariFeatureReducer):
#     """This Feature Selector creates factors from the observed variables to represent the common variance
#     i.e. variance due to correlation among the observed variables."""
#     # There are 3 steps
#     # 1) Bartlett’s Test of Sphericity and KMO Test
#     # 2) Determining the number of factors
#     # 3) Interpreting the factors
#     chi_square_value, p_value = calculate_bartlett_sphericity(df) # p_value should be <= 0.05
#     kmo_all, kmo_model = calculate_kmo(df)                        # kmo_model <= 0.6 is inadequate

#     fa = FactorAnalyzer()
#     fa.analyze(df, NUM_OF_TOTAL_VARIABLES, rotation=None)
#     ev, v = fa.get_eigenvalues()            # Check ev, pick the factors whose eigenvalues are > 1

#     fa.analyze(df, NUM_OF_FACTORS, rotation="varimax")


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