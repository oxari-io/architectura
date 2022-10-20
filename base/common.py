# from typing import Union
# import sklearn
# import numpy as np
# import pandas as pd
# from sklearn.utils.estimator_checks import check_estimator
import abc
from ctypes import Union
from pickle import pickle
import numpy as np
import pandas as pd
import sklearn
import logging
import csv
from sklearn.impute import SimpleImputer, _base


class OxariLogger:
    """
    This is the Oxari Logger class, which handles the output of any official print statement.
    The logger writes it's outputs to STDOUT or to a FILE if a LOG_FILE environment variable was set.   
    
    Task: 
    - Logger shall use a standardized prefix which provides information about the module and pipeline step
    - Logger should use an env var to determine whether to output the logging into a file or stdout
    - Avoid patterns like here https://docs.python.org/3/howto/logging-cookbook.html#patterns-to-avoid
    - In case of production env, the logger should upload the log file of the full pipeline run to digital ocean spaces
    
    """
    def __init__():
        # https://docs.python.org/3/howto/logging-cookbook.html
        pass


class OxariEvaluator(abc.ABC):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def evaluate(self, y_true, y_pred, **kwargs):
        """
        Evaluates multiple metrics and returns a dict with all computed scores.
        """
        pass

class OxariOptimizer(abc.ABC):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def optimize(self, **kwargs):
        """
        Evaluates multiple metrics and returns a dict with all computed scores.
        """
        pass

    # def compute_error_metrics(self, y_true, y_pred):
    #     """
    #     Testing the performance of the ML model
    #     Write the results in model/metrics
    #     Parameters:
    #     y_true (np.array): true value to compare predicted value
    #     y_pred (np.array): predicted value output by the model
    #     """
    #     raise NotImplementedError


class OxariMixin(abc.ABC):
    def __init__(self, object_filename, **kwargs) -> None:
        self.object_filename = object_filename
        self.start_time = None
        self.end_time = None
    

    @abc.abstractmethod
    def run(self, **kwargs) -> "OxariMixin":
        """
        Every component needs to call initialize and finish inside the run function. 
        """
        return self


    def set_logger(self, logger: OxariLogger) -> "OxariMixin":
        self._logger = logger
        return self

    def set_evaluator(self, evaluator: OxariEvaluator) -> "OxariMixin":
        self._evaluator = evaluator
        return self

    def set_optimizer(self, optimizer: OxariOptimizer) -> "OxariMixin":
        self._optimizer = optimizer
        return self



    def save_state(self):
        with open(self.object_filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_state(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class OxariTransformer(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator, abc.ABC):
    """Just for intellisense convenience. Not really necessary but allows autocompletion"""
    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariTransformer":
        return self

    @abc.abstractmethod
    def transform(self, X, kwargs) -> Union[np.ndarray, pd.DataFrame]:
        pass


class OxariClassifier(sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator, abc.ABC):
    """Just for intellisense convenience. Not really necessary but allows autocompletion"""
    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariClassifier":
        return self

    @abc.abstractmethod
    def transform(self, X, kwargs) -> Union[np.ndarray, pd.DataFrame]:
        pass


class OxariImputer(_base._BaseImputer, OxariMixin, abc.ABC):
    """
    Handles imputation of missing values for values that are zero. Fit and Transform have to be implemented accordingly.
    """
    def __init__(self, missing_values=np.nan, verbose: int = 0, copy: bool = False, add_indicator: bool = False, **kwargs):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.verbose = verbose
        self.copy = copy

    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariImputer":
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
