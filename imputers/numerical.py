from typing import Union

import numpy as np
import pandas as pd

from base.common import OxariImputer

from .core import BucketImputerBase

EXAMPLE_STRING = """
self.statistics = {
                    '0...10': {
                        'ft_numc_revenue':{
                            'min': -10,
                            'max': 1000,
                            'median': 100,
                            'mean': 450,
                        }
                    },
                    '10...20': {
                        ...
                    },
                    ...,
                }
"""


class NumericalStatisticsImputer(BucketImputerBase):

    def __init__(self, reference: str, num_buckets: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.reference = reference
        self.bucket_number = num_buckets
        self.list_of_skipped_columns = ['key_year', 'key_isin']
        # self.columns_to_fit = set(NumMapping.get_features()) - set([self.MAIN_VARIABLE])
        self.fallback_fallback_value = 0
        # TODO: Evaluate based on the entropy of buckets
        self.info = {"bucket_counts": {}}

    def _aggregations(self):
        return ["min", "max", "median", "mean"]

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "OxariImputer":
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """

        data = X.copy()
        data_numerical = data.filter(regex="ft_num", axis=1)
        min_ = data[self.reference].min()  # 20
        max_ = data[self.reference].max()  # 1000
        self.thresholds = self._get_threshold(self.bucket_number, min_, max_, data[self.reference].dropna())
        data_numerical = data_numerical.assign(
            **{self.reference: data_numerical[self.reference].fillna(data_numerical[self.reference].median()).values})
        data_numerical["bucket"] = None
        data_numerical["bucket"] = pd.cut(data[self.reference], self.thresholds)
        cat_stats: pd.DataFrame = data_numerical.groupby("bucket").aggregate(self._aggregations())
        self.stats_specific = cat_stats.to_dict(orient='index')
        self.stats_overall = {(col, key): val
                              for col in data_numerical.drop('bucket', axis=1).columns
                              for key, val in data_numerical[col].aggregate(self._aggregations()).items()}

        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = X.copy()
        X_new = X_new.to_frame() if isinstance(X_new, pd.Series) else X_new
        X_new["bucket"] = None
        X_new["bucket"] = pd.cut(X_new[self.reference], self.thresholds)

        tmp = X_new.set_index('bucket')
        for intervall_group, bgroup in tmp.filter(regex="^ft_num", axis=1).groupby("bucket"):
            # For interval group apply statistics column-wise
            filled_bgroup = bgroup.apply(lambda col: col.fillna(self.stats_specific.get(intervall_group, {}).get((col.name, 'median'))), axis=0)
            tmp.loc[tmp.index == intervall_group, filled_bgroup.columns] = filled_bgroup[filled_bgroup.columns].values
        # Apply overall statistics to fill na in every column
        tmp_filled = tmp.apply(lambda col: col.fillna(self.stats_overall.get((col.name, 'median'))) if col.name.startswith('ft_num') else col, axis=0)
        X_new[tmp_filled.columns] = tmp_filled[tmp_filled.columns].reset_index(drop=True).values
        return X_new.drop('bucket', axis=1).infer_objects()

    def _get_threshold(self, buckets_number, min_, max_, data):
        return np.linspace(min_, max_, buckets_number + 1)
    
    def __repr__(self):
        return f"@[{self.__class__.__name__}]{self.info}"

    def get_config(self, deep=True):
        return {"reference": self.reference, **super().get_config(deep)}

    @property
    def name(self):
        return self.__class__.__name__ + f":{self.reference}"



class NumericalStatisticsQuantileBucketImputer(NumericalStatisticsImputer):

    def _get_threshold(self, buckets_number, min_, max_, data):
        x = np.linspace(0, 1, buckets_number + 1)
        threshold = np.quantile(data, x)
        return threshold