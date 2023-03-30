from base.common import OxariImputer, OxariTransformer
from typing_extensions import Self
from numbers import Number

from base.oxari_types import ArrayLike
import pandas as pd

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
        for ft_col in X_new.filter(regex="^ft_num", axis=1).columns.tolist():
            for grp, replacers in stats.items():
                values_need_filling_and_are_of_grp:pd.Series = (X_new[ft_col].isna()) & (X_new[self.reference] == grp)
                if values_need_filling_and_are_of_grp.any() is not True:
                    continue
                replacement_value = replacers.get((ft_col, self.statistic)) or self.stats_overall.get((ft_col, self.statistic))
                X_new.loc[values_need_filling_and_are_of_grp, ft_col] = replacement_value
        return X_new
