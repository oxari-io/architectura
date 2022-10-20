
from typing import Union
from base import OxariScopeEstimator
import numpy as np
import pandas as pd

class MiniModelArmyEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        return super().fit(X, y, **kwargs)
    
    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        return super().predict(X)