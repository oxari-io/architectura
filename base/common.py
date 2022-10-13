
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
    @abc.abstractmethod
    def evaluate(self, y_true, y_pred):
        """
        Evaluates multiple metrics and returns a dict with all computed scores.
        """
        pass
    
    
    

class OxariMixin(abc.ABC): 
    def __init__(self, object_filename, **kwargs) -> None:
        self.object_filename = object_filename


    @abc.abstractmethod
    def run(self, **kwargs) -> "OxariMixin":
        return self
  
    
    def set_logger(self, logger: OxariLogger)-> "OxariMixin":
        self._logger = logger
        return self
    

    def set_evaluator(self, evaluator: OxariEvaluator)-> "OxariMixin":
        self._evaluator = evaluator
        return self
    
    
    @abc.abstractmethod
    def compute_error_metrics(self, y_true, y_pred):
        """
        Testing the performance of the ML model
        Write the results in model/metrics
        Parameters:
        y_true (np.array): true value to compare predicted value
        y_pred (np.array): predicted value output by the model
        """
        raise NotImplementedError
    
    
    def save_state(self):
        with open(self.object_filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_state(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class OxariTransformer(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator,  abc.ABC):
    """Just for intellisense convenience. Not really necessary but allows autocompletion"""
    
    @abc.abstractmethod
    def fit(self, X, y, **kwargs) -> "OxariTransformer":
        return self

    @abc.abstractmethod
    def transform(self, X, kwargs) -> Union[np.ndarray, pd.DataFrame]:
        pass
    
class LogarithmScaler(OxariTransformer, OxariMixin):
    def fit(self, X, y, **kwargs) -> "LogarithmScaler":
        return self

    def transform(self, X, kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return np.log1p(X)