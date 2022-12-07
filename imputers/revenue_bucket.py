import itertools
from typing import Union
from base.common import OxariImputer
from base.dataset_loader import OxariDataManager
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from base.mappings import NumMapping
from base.metrics import mape


class RevenueBucketImputer(OxariImputer):
    def __init__(self, buckets_number=3, **kwargs):
        super().__init__(**kwargs)
        self.bucket_number = buckets_number
        self.list_of_skipped_columns = ['year', 'isin'] + NumMapping.get_targets()
        self.columns_to_fit = set(NumMapping.get_features()) - set(["revenue"])
        self.fallback_fallback_value = 0
        # TODO: Evaluate based on the entropy of buckets
        self.info = {"bucket_counts": {}}

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "OxariImputer":
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """

        self.lookup_table_ = self._split_in_buckets_per_revenue(X, self.bucket_number)
        self.fallback_fallback_value = X["revenue"].median()

        # looping over lookup_table -->
        # { bucket_n : {"interval" : (0, 20), "values" : {"column" : column_mean}}}
        # where 0 and 20 are the split points (interval) of the first bucket and value represents the mean which will be used for imputing

        for bucket in self.lookup_table_.keys():

            interval = self.lookup_table_[bucket]["interval"]
            interval_overall = self.lookup_table_["default"]["interval"]

            # create a filter where the value of revenue falls in between the interval of the bucket
            filter_ = (X["revenue"].between(interval[0], interval[1], inclusive="both"))
            filter_overall = (X["revenue"].between(interval_overall[0], interval_overall[1], inclusive="both"))

            # Count how many landed in bucket
            if bucket != 'default':
                self.info["bucket_counts"][f"bucket_{bucket}"] = filter_.sum()

            data_grouped_by_revenue = X.loc[filter_]
            data_grouped_by_revenue_overall = X.loc[filter_overall]

            lookup_columns_mean = {}  # {col1 : mean1, col2 : mean2}
            lookup_columns_mean_overall = {}  # {col1 : mean1, col2 : mean2}

            for column in self.columns_to_fit:
                # This is the fallback in case the median is NaN (One company in the bucket)
                lookup_columns_mean[column] = data_grouped_by_revenue_overall[column].median()
                lookup_columns_mean_overall[column] = data_grouped_by_revenue[column].median()

            self.lookup_table_["default"]["values"] = lookup_columns_mean
            self.lookup_table_[bucket]["values"] = lookup_columns_mean_overall

        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = X.copy()
        X_new.loc[np.isnan(X_new["revenue"]), ["revenue"]] = self.fallback_fallback_value
        for bucket in self.lookup_table_.keys():
            if bucket == "default":
                continue  # Skip the default values
            interval = self.lookup_table_[bucket]["interval"]

            # check if the data to impute has revenue that falls between the intervals
            filter_ = (X_new["revenue"].between(interval[0], interval[1], inclusive="both"))

            for column in self.columns_to_fit:

                mean_of_column = self.lookup_table_[bucket]["values"][column]
                default_mean_of_column = self.lookup_table_["default"]["values"][column]
                is_na = np.isnan(X_new[column])
                if not np.isnan(mean_of_column):

                    # TODO: Log how many values are imputed by default or by bucket
                    X_new.loc[filter_ & is_na, column] = mean_of_column
                else:
                    X_new.loc[filter_ & is_na, column] = default_mean_of_column

        return X_new

    def _split_in_buckets_per_revenue(self, data, buckets_number=3):
        """
        Creates a lookup table to impute missing value based on the bucket the company is in.
        Buckets are created on revenue. 

        Outcome will be n_buckets+2 to account for both open ends
        """

        split_points = []

        min_ = data["revenue"].min()  # 20
        max_ = data["revenue"].max()  # 1000
        # TODO: Same with quantiles
        # TODO: Same with parabola
        thresholds = self._get_threshold(buckets_number, min_, max_, data["revenue"].dropna())

        split_points = [-np.inf]
        split_points.extend(list(thresholds))
        split_points.append(np.inf)

        lookup_table = {i + 1: {"interval": (split_points[i], split_points[i + 1])} for i in range(len(split_points) - 1)}
        lookup_table["default"] = {"interval": (split_points[0], split_points[-1])}

        return lookup_table

    def _get_threshold(self, buckets_number, min_, max_, data):
        return np.linspace(min_, max_, buckets_number + 1)

    def __repr__(self):
        return f"@[{self.__class__.__name__}]{self.info}"


class RevenueExponentialBucketImputer(RevenueBucketImputer):
    def _get_threshold(self, buckets_number, min_, max_, data):
        return np.geomspace(min_ - min_ + 1, max_ - min_ + 1, buckets_number + 1) + min_ - 1


class RevenueQuantileBucketImputer(RevenueBucketImputer):
    def _get_threshold(self, buckets_number, min_, max_, data):
        x = np.linspace(0, 1, buckets_number+1)
        threshold = np.quantile(data, x)
        return threshold


class RevenueParabolaBucketImputer(RevenueBucketImputer):
    def _get_threshold(self, buckets_number, min_, max_, data):
        x = np.arange(buckets_number+1)
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
