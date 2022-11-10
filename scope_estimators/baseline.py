from typing import Union
from base.pipeline import OxariScopeEstimator
import numpy as np
import pandas as pd


class DummyEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        pass
    
    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        return np.ones(len(X)) * 42

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def check_conformance(self):
        pass

    def deploy(self):
        pass


class BaselineEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        self.median_value = np.median(y)
        return self
    
    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        return np.ones(len(X)) * self.median_value 
    
    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return super().optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return super().evaluate(y_true, y_pred, **kwargs)