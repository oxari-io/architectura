from typing import Union
import sklearn
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
import abc


class BaseScopeEstimator(abc.ABC, sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, **kwargs):
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference: https://scikit-learn.org/stable/developers/develop.html#instantiation
        pass

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "BaseScopeEstimator":
        # Takes X and y and trains regressor.
        # Include If X.shape[0] == y.shape[0]: raise ValueError(f“X and y do not have the same size (f{X.shape[0]} != f{X.shape[0]})”).
        # Set self.n_features_in_ = X.shape[1]
        # Avoid setting X and y as attributes. Only increases the model size.
        # When fit is called, any previous call to fit should be ignored.
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore. (e.g.: self.coef_)
        # Reference: https://scikit-learn.org/stable/developers/develop.html#fitting
        return self

    @abc.abstractmethod
    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        # Takes X and computes predictions
        # Returns prediction results
        pass

    @abc.abstractmethod
    def check_conformance(self, ) -> bool:
        # Returns a boolean
        # Uses sklearn utils function check_estimator(self)
        # If this test passes, then deployment shall be allowed
        # check_estimator Makes sure that we can use model evaluation and selection tools such as model_selection.GridSearchCV and pipeline.Pipeline.
        # Reference
        pass

    @abc.abstractmethod
    def deploy(self, ) -> bool:
        # pickles and deploys the new models; multiple options are
        # possible here:
        # upload them in DigitalOcean Spaces
        # dockerize and deploy to scalable DigitalOcean droplet
        # extract the code to a new repository and add it to the current DigitalOcean App as a new component
        # keep it inside the current backend directly which would require more RAM
        pass
