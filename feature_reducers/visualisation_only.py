import pandas as pd
from sklearn.manifold import (MDS, SpectralEmbedding)
from typing_extensions import Self
from base.common import OxariFeatureReducer

class MDSDimensionalityFeatureReducer(OxariFeatureReducer):
    """ This Feature Selector uses Multidimensional Scaling
    You can find an explanation here: https://www.statisticshowto.com/multidimensional-scaling/ 
    """

    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = MDS(n_components=n_components, normalized_stress='auto')

    # "Compute the embedding vectors for data X."
    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.feature_names_in_ = list(X.filter(regex="^ft_", axis=1).columns)
        self.n_components_ = self._dimensionality_reducer.n_components
        self.logger.info(f'Number of components before dimensionality reduction: {len(self.feature_names_in_)}')
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        X_new = X.copy()
        X_reduced = pd.DataFrame(self._dimensionality_reducer.fit_transform(X_new[self.feature_names_in_]), index=X_new.index)
        X_new_reduced = self.merge(X_new, X_reduced)
        self.logger.info(f'Number of components after dimensionality reduction: {len(X_new_reduced.columns)}')
        # fit_transform will generate a new embedding space instead of projecting new points
        # into the same embedding space used for the reference data.
        # Plus the doc-page indicates that the output is an ndarray, not an array-like object (tho that's not an issue???)
        return X_new_reduced


class SpectralEmbeddingFeatureReducer(MDSDimensionalityFeatureReducer):
    """This Feature Selector finds a low dimensional representation of the data using 
    a spectral decomposition of the graph Laplacian"""

    def __init__(self, n_components=5, **kwargs):
        super().__init__(**kwargs)
        self._dimensionality_reducer = SpectralEmbedding(n_components=n_components)