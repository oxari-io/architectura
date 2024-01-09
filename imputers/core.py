from typing import Union
from numpy import nan
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from typing_extensions import Self
import numpy as np
import pandas as pd
# noqa
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

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

class RegressionImputerBase(OxariImputer):
    def __init__(self, internal_scaler=PowerTransformer(), **kwargs):
        super().__init__(**kwargs)
        self._scaler = internal_scaler
        self._estimator = IterativeImputer(estimator=LinearRegression(), verbose=self.verbose)

    def _fit_scaler(self, X:pd.DataFrame, y=None):
        """
        Trains the internal scaler for this imputer. Returns scaled version of the training data.
        """
        X_train = X
        self._scaler = self._scaler.fit(X_train)
        X_train_scaled = self._scaler.transform(X_train)
        return pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    
    def _scale_transform(self, X_num:pd.DataFrame):
        return self._estimator.transform(pd.DataFrame(self._scaler.transform(X_num), index=X_num.index, columns=X_num.columns))    