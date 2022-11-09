from typing import Union
from base.pipeline import OxariScopeEstimator
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

STANDARD_KERNEL = DotProduct() + WhiteKernel()
class GaussianProcessEstimator(OxariScopeEstimator):
    def __init__(self, kernel=STANDARD_KERNEL, **kwargs):
        super().__init__(**kwargs)
        self._gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        self._gpr =  self._gpr.fit(X, y, **kwargs)
        return self
    
    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        return self._gpr.predict(X)

    def check_conformance(self):
        pass

    def deploy(self):
        pass
