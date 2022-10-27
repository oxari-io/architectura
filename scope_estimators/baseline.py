from typing import Union
from base.pipeline import OxariScopeEstimator
import numpy as np
import pandas as pd


class DefaultScopeEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        return super().fit(X, y, **kwargs)
    
    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        return super().predict(X)

    def check_conformance(self):
        pass

    def deploy(self):
        pass
