import itertools
from typing import Union

import numpy as np
import pandas as pd

from base.common import OxariImputer
from base.mappings import NumMapping


class ScopeBucketImputer(OxariImputer):
    def __init__(self, buckets_number = 3, **kwargs):
        # self._imputer = SimpleImputer(missing_values, verbose, copy, add_indicator, **kwargs)
        super().__init__(**kwargs)
        self.bucket_number = buckets_number
        #self.list_of_skipped_columns = ['year', 'isin', "ticker", "target"]
        self.list_of_skipped_columns = ['year', 'isin'] + NumMapping.get_targets()
        self.columns_to_fit = NumMapping.get_features()

        
    def fit(self, X:pd.DataFrame, y, **kwargs) -> "OxariImputer":
        """
        Computes a Lookup Matrix with bucket splits and feature
        """
        # TODO: generate attribute for imputing missing values for all the columns based on sector name and country
        self._split_in_buckets(X)

        # shady solution over here, scopes have -1 instead of nan so they dont get filled
        X.loc[ : , NumMapping.get_targets()] = X[NumMapping.get_targets()].replace(np.nan, -1)

        for bs1, bs2 in itertools.product(self.buckets_dict_s1.keys(), self.buckets_dict_s2.keys()):
            interval_s1 = self.buckets_dict_s1[bs1]
            interval_s2 = self.buckets_dict_s2[bs2]

            filter_ = (X["scope_1"].between(interval_s1[0], interval_s1[1], inclusive = "left")) & (X["scope_2"].between(interval_s2[0], interval_s2[1], inclusive = "left"))

            # this will not fill the scopes bc scopes have -1 instead of na
            X.loc[filter_] = X.loc[filter_].fillna(X.loc[filter_].mean())

        # filling remaining gaps with overall mean
        X.loc[ : , NumMapping.get_features()] = X[NumMapping.get_features()].fillna(X[NumMapping.get_features()].mean())        
        return self
    
    def transform(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        # using sector specific median since using global mean would be dumb
        for col in self.columns_to_fit:
            X[col] = X.groupby("sector_name")[col].apply(lambda x: x.fillna(x.median()))
            # some values dont get filled after the line above, maybe bc of "sector_name"?
            # therefore we used the global mean of the columns to fill up the remaning nans
            X[col].fillna(X[col].mean(), inplace = True)
        
        return X
    

    @staticmethod
    def _split_in_buckets_per_scope(data, scope, buckets_number = 3):

        """
        scope has to be string-like --> "scope_i" where i is in [1,2]
        """

        split_points = []

        min_ = max(data[scope].min(), 0) # 20
        max_ = data[scope].max() # 1000
        
        full_intervall = max_ - min_ # 980
        offset = full_intervall /  buckets_number # 980/3 = 326
        
        multiplier = np.arange(0, buckets_number-1)
        split_points = (multiplier * offset) + min_ # (0+20, 326+20, 652+20) 
        
        # NOTE: Needs to be fully numpyified
        split_points = list(split_points)
        split_points.append(max_)
        
        # split_points.append(min_)
        # for i in range(buckets_number-1):
            # split_points.append( ( (i+1) * offset) + min_)
        # split_points.append(max_)
        # split_points = [20, 346, 672, 1000]

        buckets_dict = { i+1 : (split_points[i], split_points[i+1]) for i in range(buckets_number)}


        return buckets_dict

    def _split_in_buckets(self, data):

        self.buckets_dict_s1 = BucketImputer._split_in_buckets_per_scope(data, "scope_1", self.buckets_number)

        self.buckets_dict_s2 = BucketImputer._split_in_buckets_per_scope(data, "scope_2", self.buckets_number)

    