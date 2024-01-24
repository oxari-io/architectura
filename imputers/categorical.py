from base.common import OxariImputer, OxariTransformer
from typing_extensions import Self
from numbers import Number

from base.oxari_types import ArrayLike
import pandas as pd
import numpy as np 

EXAMPLE_STRING = """
self.statistics = {
                'country_code': {
                    'GER': {
                        'min': -10,
                        'max': 1000,
                        'median': 100,
                        'mean': 450,
                    },
                    'USA': {
                        'min': 0,
                        'max': 300,
                        'median': 100,
                        'mean': 250,
                    },
                    ...,
                },
                'industry_name': {
                    'Beverages': {
                        'min': -10,
                        'max': 1000,
                        'median': 100,
                        'mean': 450,
                    },
                    'Automotive': {
                        'min': 0,
                        'max': 300,
                        'median': 100,
                        'mean': 250,
                    },
                    ...,
                },
                ...,
            }
"""

COL_REFERENCE = "ft_catm_country_code"
MappingType = dict[tuple[str, str], Number] 

class CategoricalStatisticsImputer(OxariImputer):

    def __init__(self, reference=COL_REFERENCE, statistic='median', name=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.reference = reference
        self.statistic = statistic
        self.stats_specific:dict[str, dict[str, MappingType]] = {}
        self.stats_overall:MappingType = {}

    def _aggregations(self):
        return ["min", "max", "median", "mean"]

    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        f"""
        Gathers statistics in the form of:
        {EXAMPLE_STRING}
        """
        X_new = X.copy()
        for col in X_new.filter(regex="^ft_cat", axis=1).columns:
            other_categoricals = set(X_new.filter(regex="^(ft_cat|key)").columns)-set([col])
            cat_stats: pd.DataFrame = X_new.drop(other_categoricals, axis=1).groupby(col).aggregate(self._aggregations())
            self.stats_specific[col] = cat_stats.to_dict(orient='index')
        self.stats_overall = {(col, key): val for col in X.filter(regex="^ft_num", axis=1).columns for key, val in X_new[col].aggregate(self._aggregations()).items()}
        return self

    def transform(self, X: ArrayLike | pd.Series, **kwargs) -> ArrayLike:
        X_new = X.copy()
        X_new = X_new.to_frame() if isinstance(X_new, pd.Series) else X_new
        stats = self.stats_specific[self.reference]
        all_columns = X_new.filter(regex="^ft_num", axis=1).columns.tolist()
        for ft_col in all_columns:
            for value, replacers in stats.items():
                values_need_filling_and_are_of_grp:pd.Series = (X_new[ft_col].isna()) & (X_new[self.reference] == value)
                num_values_to_replace = len(X_new.loc[values_need_filling_and_are_of_grp, ft_col])
                if not num_values_to_replace:
                    continue
                specific_replacement_value = replacers.get((ft_col, self.statistic))
                overall_replacement_value = self.stats_overall.get((ft_col, self.statistic))
                replacement_value = specific_replacement_value if not (np.isnan(specific_replacement_value) or specific_replacement_value is None) else overall_replacement_value
                X_new.loc[values_need_filling_and_are_of_grp, ft_col] = replacement_value
        for ft_col in all_columns:
            # All whose values are still none necause the reference itself is None.
            reference_value_is_none = (X_new[ft_col].isna()) & (X_new[self.reference].isna())
            num_values_to_replace = len(X_new.loc[reference_value_is_none, ft_col])
            if not num_values_to_replace:
                continue
            X_new.loc[reference_value_is_none, ft_col] = self.stats_overall.get((ft_col, self.statistic))
        for ft_col in all_columns:
            # All whose values are still none because not enough statistics where gathered. (Some values don't exist.)
            remaining = (X_new[ft_col].isna())
            num_values_to_replace = len(X_new.loc[remaining, ft_col])
            if not num_values_to_replace:
                continue
            X_new.loc[remaining, ft_col] = self.stats_overall.get((ft_col, self.statistic))
        return X_new

    def get_config(self, deep=True):
        return {"reference": self.reference, "imputer": f"{self.name}:{self.reference}", **super().get_config(deep)}
    
    
class HybridCategoricalStatisticsImputer(CategoricalStatisticsImputer):
    
    def __init__(self, statistic='median', name=None, **kwargs) -> None:
        super().__init__(("ft_catm_country_code", "ft_catm_industry_name"), statistic, name, **kwargs)
        
    def fit(self, X: ArrayLike, y: ArrayLike = None, **kwargs) -> Self:
        X_new = X.copy()
        col = list(self.reference)
        other_categoricals = set(X_new.filter(regex="^(ft_cat|key)").columns)-set(col)
        cat_stats: pd.DataFrame = X_new.drop(other_categoricals, axis=1).groupby(col).aggregate(self._aggregations())
        self.stats_specific[self.reference] = cat_stats.to_dict(orient='index')
        

            
        self.stats_overall = {(col, key): val for col in X.filter(regex="^ft_num", axis=1).columns for key, val in X_new[col].aggregate(self._aggregations()).items()}
        return self
    
    
    def transform(self, X: ArrayLike | pd.Series, **kwargs) -> ArrayLike:
        X_new = X.copy()
        X_new = X_new.to_frame() if isinstance(X_new, pd.Series) else X_new
        stats = self.stats_specific[self.reference]
        all_columns = X_new.filter(regex="^ft_num", axis=1).columns.tolist()
        for ft_col in all_columns:
            for value, replacers in stats.items():
                values_need_filling_and_are_of_grp:pd.Series = (X_new[ft_col].isna()) & (np.all(X_new[list(self.reference)] == list(value), axis=1))
                num_values_to_replace = len(X_new.loc[values_need_filling_and_are_of_grp, ft_col])
                if not num_values_to_replace:
                    continue
                specific_replacement_value = replacers.get((ft_col, self.statistic))
                overall_replacement_value = self.stats_overall.get((ft_col, self.statistic))
                replacement_value = specific_replacement_value if not (np.isnan(specific_replacement_value) or specific_replacement_value is None) else overall_replacement_value
                X_new.loc[values_need_filling_and_are_of_grp, ft_col] = replacement_value
        for ft_col in all_columns:
            # All whose values are still none necause the reference itself is None.
            reference_value_is_none = (X_new[ft_col].isna()) & (X_new[list(self.reference)].isna().any(axis=1))
            num_values_to_replace = len(X_new.loc[reference_value_is_none, ft_col])
            if not num_values_to_replace:
                continue
            X_new.loc[reference_value_is_none, ft_col] = self.stats_overall.get((ft_col, self.statistic))
        for ft_col in all_columns:
            # All whose values are still none because not enough statistics where gathered. (Some values don't exist.)
            remaining = (X_new[ft_col].isna())
            num_values_to_replace = len(X_new.loc[remaining, ft_col])
            if not num_values_to_replace:
                continue
            X_new.loc[remaining, ft_col] = self.stats_overall.get((ft_col, self.statistic))
        return X_new    