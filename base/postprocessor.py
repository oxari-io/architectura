from typing import Union
import sklearn
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
import abc
from base import common 

class OxariPostprocessor(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, common.OxariMixin, abc.ABC):
    def __init__(self, **kwargs):
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference: https://scikit-learn.org/stable/developers/develop.html#instantiation
        pass

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariPostprocessor":
        # Takes X and y and trains regressor.
        # Include If X.shape[0] == y.shape[0]: raise ValueError(f“X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})”).
        # Set self.n_features_in_ = X.shape[1]
        # Avoid setting X and y as attributes. Only increases the model size.
        # When fit is called, any previous call to fit should be ignored.
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        # Reference: https://scikit-learn.org/stable/developers/develop.html#fitting
        return self


    @abc.abstractmethod
    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        pass