from typing import Union
from base.common import OxariImputer
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class DummyImputer(OxariImputer):
    def __init__(self, strategy="median", missing_values=np.nan, verbose=0, copy=True, add_indicator=False, **kwargs):
        self._imputer = SimpleImputer(
            missing_values=missing_values,
            strategy=strategy,
            verbose=verbose,
            copy=copy,
            add_indicator=add_indicator,
            **kwargs,
        )

    def fit(self, X, y=None, **kwargs) -> "OxariImputer":
        self._imputer.fit(X, y, **kwargs)
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        return self._imputer.transform(X, **kwargs)