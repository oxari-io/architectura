from typing import  Union
from typing_extensions import Self
import numpy as np
import pandas as pd

from base.common import OxariImputer
from base.mappings import NumMapping
from base.oxari_types import ArrayLike
from abc import abstractmethod

from imputers.categorical import CategoricalStatisticsImputer

COL_GROUPER = 'key_isin'
COL_TIME = 'key_year'
class InterpolationImputer(OxariImputer):
    def __init__(self, method:str='linear', window_size:int=3, **kwargs):
        super().__init__(**kwargs)
        self.method=method
        self.window_size=window_size
        self.fallback_imputer = CategoricalStatisticsImputer()

    def fit(self, X:ArrayLike, y=None, **kwargs) -> Self:
        self.fallback_imputer.fit(X, y)
        return super().fit(X, y, **kwargs)

    def _transform_col(self, col:pd.Series):
        tmp_col = col.copy()
        col_not_na = ~tmp_col.isna()
        num_not_na = col_not_na.sum()
        if num_not_na == len(tmp_col):
            return tmp_col
        if num_not_na <= 2:
            return tmp_col
        # if num_not_na == 1:
        #     tmp_col = tmp_col.bfill().ffill()
        #     return tmp_col

        # fill in the missing values in between the first and last non-null values
        start_idx = tmp_col.first_valid_index()
        end_idx = tmp_col.last_valid_index()
        if (end_idx - start_idx)+1 < self.window_size:
            return tmp_col

        s_new = tmp_col.interpolate(self.method, limit_area=None)


        # fill in the missing values backward with a moving average
        # As backward fill is the same as forward fill reversed we use [::-1] to reverse the sequence and [::-1] to turn it to original sequence direction
        s_new = self._rolling_mean_fill(s_new[::-1])[::-1]

        # fill in the missing values forward with a moving average
        s_new = self._rolling_mean_fill(s_new)
        s_new = s_new.astype(col.dtype)
        return s_new

    def _rolling_mean_fill(self, series:pd.Series)->pd.Series:
        tmp = series.copy()
        end_idx = tmp.reset_index(drop=True).last_valid_index()
        # tmp
        while tmp.iloc[end_idx:].isna().any():
            tmp.iloc[end_idx+1] = tmp.iloc[np.maximum(end_idx-self.window_size, 0):end_idx+1].mean()
            end_idx = tmp.reset_index(drop=True).last_valid_index()
        return tmp

    def _interpolate(self, group:pd.DataFrame):
        X = group.copy()
        tmp_df = X.set_index(COL_TIME).drop([COL_GROUPER], axis=1).transform(self._transform_col)
        # result = tmp_df.interpolate(self.method)
        X[tmp_df.columns] = tmp_df[tmp_df.columns].values
        X = X.infer_objects()
        return X

    def transform(self, X:ArrayLike, **kwargs) -> ArrayLike:
        result = X.sort_values([COL_GROUPER, COL_TIME]).drop_duplicates([COL_GROUPER, COL_TIME]).filter(regex="^(?!tg_).*", axis=1).groupby(COL_GROUPER, group_keys=False).progress_apply(self._interpolate)
        result = self.fallback_imputer.transform(result)
        return result 

class LinearInterpolationImputer(InterpolationImputer):
    def __init__(self, window_size: int = 3, **kwargs):
        super().__init__('linear', window_size, **kwargs)


class SplineInterpolationImputer(InterpolationImputer):
    def __init__(self, window_size: int = 3, **kwargs):
        super().__init__('cubicspline', window_size, **kwargs)


