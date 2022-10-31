import itertools
from typing import Union
from base.common import OxariImputer
from base.dataset_loader import OxariDataLoader
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

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "OxariImputer":
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """

        self.lookup_table_ = self._split_in_buckets_per_revenue(X)

        # looping over lookup_table -->
        # { bucket_n : {"interval" : (0, 20), "values" : {"column" : column_mean}}}
        # where 0 and 20 are the split points (interval) of the first bucket and value represents the mean which will be used for imputing
        for bucket in self.lookup_table_.keys():

            interval = self.lookup_table_[bucket]["interval"]

            # create a filter where the value of revenue falls in between the interval of the bucket
            filter_ = (X["revenue"].between(interval[0], interval[1], inclusive="both"))

            data_grouped_by_revenue = X.loc[filter_]

            lookup_columns_mean = {}  # {col1 : mean1, col2 : mean2}

            for column in self.columns_to_fit:

                # getting the mean of the column for the data grouped in buckets by revenue
                mean_of_column = data_grouped_by_revenue[column].median()  

                # assign the mean value to the lookup dictionary
                lookup_columns_mean[column] = mean_of_column

                self.lookup_table_[bucket]["values"] = lookup_columns_mean

        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:

        for bucket in self.lookup_table_.keys():

            interval = self.lookup_table_[bucket]["interval"]

            try:
                # check if the data to impute has revenue that falls between the intervals
                filter_ = (X["revenue"].between(interval[0], interval[1], inclusive="left"))

                for column in self.columns_to_fit:

                    mean_of_column = self.lookup_table_[bucket]["values"][column]

                    # REVIEWME: is the syntax below correct?
                    X.loc[filter_, column].fillna(mean_of_column, inplace=True)

            except:
                # in case the data to impute does not have a revenue value that falls between the interval then go to the next iteration
                print("Something went massively wrong in REVENUE_IMPUTER")

        return X

    @staticmethod
    def _split_in_buckets_per_revenue(data, buckets_number=3):
        """
        Creates a lookup table to impute missing value based on the bucket the company is in.
        Buckets are created on revenue.

        # REVIEWME: Maybe fill up the missing revenue with global mean before? If not, how to handle data that does not have revenue?
        """

        split_points = []

        min_ = max(data["revenue"].min(), 0)  # 20
        max_ = data["revenue"].max()  # 1000

        full_intervall = max_ - min_  # 980
        offset = full_intervall / buckets_number  # 980/3 = 326

        multiplier = np.arange(0, buckets_number)
        split_points = (multiplier * offset) + min_  # (0+20, 326+20, 652+20)

        # NOTE: Needs to be fully numpyified
        split_points = list(split_points)
        split_points.append(max_)

        lookup_table = {i + 1: {"interval": (split_points[i], split_points[i + 1])} for i in range(buckets_number)}

        return lookup_table
