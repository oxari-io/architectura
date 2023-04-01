from typing import Union

import numpy as np
import pandas as pd

from base.common import OxariImputer

from .core import BucketImputerBase

MAIN_VARIABLE = 'ft_numc_revenue'
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


class RevenueBucketImputer(BucketImputerBase):

    def __init__(self, buckets_number: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.bucket_number = buckets_number
        self.list_of_skipped_columns = ['key_year', 'key_isin']
        # self.columns_to_fit = set(NumMapping.get_features()) - set([MAIN_VARIABLE])
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
        min_ = data[MAIN_VARIABLE].min()  # 20
        max_ = data[MAIN_VARIABLE].max()  # 1000
        self.thresholds = self._get_threshold(self.bucket_number, min_, max_, data[MAIN_VARIABLE].dropna())
        data_numerical[MAIN_VARIABLE] = data_numerical[MAIN_VARIABLE].fillna(data_numerical[MAIN_VARIABLE].median())
        data_numerical["bucket"] = pd.cut(data[MAIN_VARIABLE], self.thresholds)
        cat_stats: pd.DataFrame = data_numerical.groupby("bucket").aggregate(self._aggregations())
        self.stats_specific = cat_stats.to_dict(orient='index')
        self.stats_overall = {(col, key): val for col in data_numerical.drop('bucket', axis=1).columns for key, val in data_numerical[col].aggregate(self._aggregations()).items()}

        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = X.copy()
        X_new = X_new.to_frame() if isinstance(X_new, pd.Series) else X_new
        X_new["bucket"] = pd.cut(X_new[MAIN_VARIABLE], self.thresholds)

        tmp = X_new.set_index('bucket')
        for intervall_group, bgroup in tmp.filter(regex="^ft_num", axis=1).groupby("bucket"):
            # For interval group apply statistics column-wise
            filled_bgroup = bgroup.apply(lambda col: col.fillna(self.stats_specific.get(intervall_group, {}).get((col.name, 'median'))),axis=0)
            tmp.loc[tmp.index==intervall_group, filled_bgroup.columns]=filled_bgroup[filled_bgroup.columns].values
        # Apply overall statistics to fill na in every column
        tmp_filled = tmp.apply(lambda col: col.fillna(self.stats_overall.get((col.name, 'median'))),axis=0)
        X_new[tmp_filled.columns] = tmp_filled[tmp_filled.columns].reset_index(drop=True).values
        return X_new.drop('bucket', axis=1).infer_objects()

    def _get_threshold(self, buckets_number, min_, max_, data):
        return np.linspace(min_, max_, buckets_number + 1)

    def __repr__(self):
        return f"@[{self.__class__.__name__}]{self.info}"


class RevenueExponentialBucketImputer(RevenueBucketImputer):

    def _get_threshold(self, buckets_number, min_, max_, data):
        return np.geomspace(min_ - min_ + 1, max_ - min_ + 1, buckets_number + 1) + min_ - 1


class RevenueQuantileBucketImputer(RevenueBucketImputer):

    def _get_threshold(self, buckets_number, min_, max_, data):
        x = np.linspace(0, 1, buckets_number + 1)
        threshold = np.quantile(data, x)
        return threshold


class RevenueParabolaBucketImputer(RevenueBucketImputer):

    def _get_threshold(self, buckets_number, min_, max_, data):
        x = np.arange(buckets_number + 1)
        start, middle, stop = x[0], x[buckets_number // 2], x[-1]
        A, B, C = self.calc_parabola_vertex(start, min_, middle, 0, stop, max_)
        return A * (x**2) + B * x + C

    def calc_parabola_vertex(self, x1, y1, x2, y2, x3, y3):
        '''
        Adapted and modifed to get the unknowns for defining a parabola:
        http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
        Taken from http://chris35wills.github.io/parabola_python/
        '''

        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

        return A, B, C
