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

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "OxariImputer":
        """
        Creates a lookup table to impute missing values based on the buckets created on revenue
        """

        self.lookup_table_ = self._split_in_buckets_per_revenue(X)
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
            data_grouped_by_revenue = X.loc[filter_]
            data_grouped_by_revenue_overall = X.loc[filter_overall]

            lookup_columns_mean = {}  # {col1 : mean1, col2 : mean2}
            lookup_columns_mean_overall = {}  # {col1 : mean1, col2 : mean2}

            for column in self.columns_to_fit:

                # getting the mean of the column for the data grouped in buckets by revenue
                # mean_of_column =   

                # assign the mean value to the lookup dictionary
                # if np.isnan(mean_of_column):
                    # This is the fallback in case the median is NaN (One company in the bucket)
                lookup_columns_mean[column] = data_grouped_by_revenue_overall[column].median()
                lookup_columns_mean_overall[column] = data_grouped_by_revenue[column].median()
                # else:
                #     lookup_columns_mean[column] = mean_of_column
                #     self.lookup_table_[bucket]["values"] = lookup_columns_mean
                    
            self.lookup_table_["default"]["values"] = lookup_columns_mean
            self.lookup_table_[bucket]["values"] = lookup_columns_mean_overall

                
            
        return self

    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        X_new = X.copy()
        X_new.loc[np.isnan(X_new["revenue"]),["revenue"]] = self.fallback_fallback_value
        for bucket in self.lookup_table_.keys():
            if bucket == "default":
                continue # Skip the default values
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
                if X_new.loc[filter_ & is_na, column].isna().sum().sum() > 0:
                    print("Shit")
                    raise Exception('Not all values are imputed!')   



        return X_new

    @staticmethod
    def _split_in_buckets_per_revenue(data, buckets_number=3):
        """
        Creates a lookup table to impute missing value based on the bucket the company is in.
        Buckets are created on revenue.

        # REVIEWME: Maybe fill up the missing revenue with global mean before? If not, how to handle data that does not have revenue?
        """

        split_points = []

        min_ = data["revenue"].min()  # 20
        max_ = data["revenue"].max()  # 1000

        full_intervall = max_ - min_  # 980
        offset = full_intervall / buckets_number  # 980/3 = 326

        
        multiplier = np.arange(0, buckets_number)
        split_points = (multiplier * offset) + min_  # (0+20, 326+20, 652+20)

        # NOTE: Needs to be fully numpyified
        split_points = [-np.inf]
        split_points.extend(list(split_points))
        split_points.append(max_)
        split_points.append(np.inf)

        lookup_table = {i + 1: {"interval": (split_points[i], split_points[i + 1])} for i in range(buckets_number)}
        lookup_table["default"] = {"interval": (split_points[0], split_points[-1])}
        

        return lookup_table
