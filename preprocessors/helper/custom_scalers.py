
from typing import Union
import numpy as np
import pandas as pd
from base.common import OxariTransformer, OxariMixin

class LogarithmScaler(OxariTransformer):
    def fit(self, X, y=None, **kwargs) -> "LogarithmScaler":
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return np.log1p(X)
    
    
