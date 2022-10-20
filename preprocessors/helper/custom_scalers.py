
from ctypes import Union
import numpy as np
import pandas as pd
from base.common import OxariTransformer, OxariMixin

class LogarithmScaler(OxariTransformer, OxariMixin):
    def fit(self, X, y, **kwargs) -> "LogarithmScaler":
        return self

    def transform(self, X, kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return np.log1p(X)