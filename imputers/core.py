from typing import Union
from typing_extensions import Self
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from base.common import OxariImputer
from base.helper import replace_ft_num
from base.oxari_types import ArrayLike


class BaselineImputer(OxariImputer):

    def __init__(self, strategy="median", missing_values=np.nan, copy=True, add_indicator=False, **kwargs):
        super().__init__(**kwargs)
        self._imputer = SimpleImputer(
            missing_values=missing_values,
            strategy=strategy,
            # verbose=verbose,
            copy=copy,
            add_indicator=add_indicator,
            **kwargs,
        )

    def fit(self, X, y=None, **kwargs) -> Self:
        self._imputer.fit(X.filter(regex="^ft_num"), y, **kwargs)
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        X_num = X.filter(regex="^ft_num")
        X_new = self._imputer.transform(X_num)
        return replace_ft_num(X, X_new)


class DummyImputer(OxariImputer):

    def __init__(self, missing_values=np.nan, copy=True, add_indicator=False, **kwargs):
        super().__init__(**kwargs)
        self._imputer = SimpleImputer(
            missing_values=missing_values,
            strategy='constant',
            # verbose=verbose,
            fill_value=0,
            copy=copy,
            add_indicator=add_indicator,
            **kwargs,
        )

    def fit(self, X, y=None, **kwargs) -> Self:
        self._imputer.fit(X.filter(regex="^ft_num"), y, **kwargs)
        return self

    def transform(self, X, **kwargs) -> ArrayLike:
        X_num = X.filter(regex="^ft_num")
        X_new = self._imputer.transform(X_num)
        return replace_ft_num(X, X_new)


class BucketImputerBase(OxariImputer):
    bucket_number: int = None

    def get_config(self, deep=True):
        return {"bucket_number": self.bucket_number, "imputer": f"{self.name}:{self.bucket_number}-buckets", **super().get_config(deep)}
