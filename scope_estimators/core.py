from typing import Union
from base import OxariScopeEstimator
import numpy as np
import pandas as pd
from base.oxari_types import ArrayLike

class DummyEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        self.value = 42
        return self
    
    def predict(self, X) -> ArrayLike:
        return np.ones(len(X)) * self.value


class BaselineEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        self.low = np.min(y)
        self.high = np.max(y)
        return self
    
    def predict(self, X) -> ArrayLike:
        return np.random.uniform(self.low, self.high, len(X))
    

class PredictMedianEstimator(DummyEstimator):        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        y_ = np.random.choice(y,len(y)//4, replace=False)
        self.value = np.median(y_)
        return self

class PredictLowerQuartileEstimator(DummyEstimator):
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        y_ = np.random.choice(y,len(y)//4, replace=False)
        self.value = np.quantile(y_, 0.25)
        return self

class PredictUpperQuartileEstimator(DummyEstimator):
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        y_ = np.random.choice(y,len(y)//4, replace=False)
        self.value = np.quantile(y_, 0.75)
        return self
        

class PredictMeanEstimator(DummyEstimator):        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        y_ = np.random.choice(y,len(y)//4, replace=False)        
        self.value = np.mean(y_)
        return self
    


