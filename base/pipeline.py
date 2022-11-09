from typing import Union, Dict
import sklearn
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
import abc

from base.dataset_loader import OxariDataLoader
from base import common


class OxariPreprocessor(common.OxariTransformer, common.OxariMixin, abc.ABC):
    def __init__(self, imputer: common.OxariImputer = None, **kwargs):
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference:  https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.imputer = imputer

    @abc.abstractmethod
    def fit(self, X, y=None, **kwargs) -> "OxariPreprocessor":
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

    def set_imputer(self, imputer: common.OxariImputer) -> "OxariPreprocessor":
        self.imputer = imputer
        return self

    # def set_feature_selector(self, feature_selector: common.OxariFeatureSelector) -> "OxariPreprocessor":
    #     self.feature_selector = feature_selector
    #     return self


class OxariScopeEstimator(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, common.OxariMixin, abc.ABC):
    def __init__(self, **kwargs):
        # Only data independant hyperparams.
        # Hyperparams only as keyword arguments
        # Does not contain any logic except setting hyperparams immediately as class attributes
        # Reference: https://scikit-learn.org/stable/developers/develop.html#instantiation
        pass

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
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



class OxariPostprocessor(common.OxariTransformer, common.OxariMixin, abc.ABC):
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


class OxariFeatureSelector(sklearn.base.TransformerMixin, common.OxariMixin, abc.ABC):
    """
    Handles removal of unimportant features. Fit and Transform have to be implemented accordingly.
    """
    def __init__(self, missing_values=np.nan, verbose: int = 0, copy: bool = False, add_indicator: bool = False, **kwargs):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.verbose = verbose
        self.copy = copy

    @abc.abstractmethod
    def fit(self, X, y=None, **kwargs) -> "OxariFeatureSelector":
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


# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
class OxariPipeline(sklearn.base.MetaEstimatorMixin, abc.ABC):
    def __init__(
        self,
        dataset: OxariDataLoader = None,
        preprocessor: OxariPreprocessor = None,
        feature_selector: OxariFeatureSelector = None,
        scope_estimator: OxariScopeEstimator = None,
        postprocessor: OxariPostprocessor = None,
        database_deployer=None,
    ):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.scope_estimator = scope_estimator
        self.postprocessor = postprocessor
        self._start_time = None
        self._end_time = None
        # self.resources_postprocessor = database_deployer

    @abc.abstractmethod
    def run_pipeline(self, **kwargs):
        # load dataset and hold in class
        #  dataset
        pass
    
    @abc.abstractmethod
    def predict(self, **kwargs):
        pass


class OxariModel(common.OxariRegressor, common.OxariMixin, sklearn.base.MultiOutputMixin, abc.ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipelines:Dict[str, OxariPipeline] = {}
    
    def add_pipeline(self, scope:int, pipeline:OxariPipeline) -> "OxariModel":
        # TODO: Implement failsafe to not exceed the limit of three scopes
        if not isinstance(scope, int):
            raise Exception("scope is not an int")     
        self.pipelines[f"scope_{scope}"] = pipeline
        return self
    
    def get_estimator(self, scope:int) -> OxariPipeline:
        return self.pipelines[f"scope_{scope}"]
    
    def fit(self, X, y=None, **kwargs):
        pass
    
    def predict(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        scope = kwargs.get("scope", "all")
        if scope == "all":
            return self._predict_all(X, **kwargs)
        return self.get_estimator(scope).predict(X, **kwargs) 
    
    def _predict_all(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        all_predictions = []
        for _, estimator in self.pipelines.items():
            y_pred = estimator.predict(X, **kwargs)
            all_predictions.append(y_pred)
        result = np.hstack(all_predictions)
        return result