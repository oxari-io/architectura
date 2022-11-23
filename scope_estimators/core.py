from typing import Union
from base import OxariScopeEstimator
import numpy as np
import pandas as pd
from base.oxari_types import ArrayLike

class DummyEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        return self
    
    def predict(self, X) -> ArrayLike:
        return np.ones(len(X)) * 42

    def check_conformance(self):
        pass



class BaselineEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        self.low = np.min(y)
        self.high = np.max(y)
        return self
    
    def predict(self, X) -> ArrayLike:
        return np.random.uniform(self.low, self.high, len(X))
    
    def check_conformance(self):
        pass




class PredictMedianEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        self.median_value = np.median(y)
        return self
    
    def predict(self, X) -> ArrayLike:
        return np.ones(len(X)) * self.median_value 
    
    def check_conformance(self):
        pass




class PredictMeanEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        self.mean_value = np.mean(y)
        return self
    
    def predict(self, X) -> ArrayLike:
        return np.ones(len(X)) * self.mean_value 
    
    def check_conformance(self):
        pass

