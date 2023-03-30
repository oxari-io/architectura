import numpy as np
import pandas as pd
from typing_extensions import Self

from pandas.core.computation.expressions import evaluate

from base import OxariMetaModel, OxariPostprocessor
import tqdm

from base.common import OxariEvaluator, OxariImputer, OxariLoggerMixin, OxariTransformer
from base.oxari_types import ArrayLike

tqdm.tqdm.pandas()


class MissingYearImputer(OxariImputer):
    COL_GROUP = 'key_isin'
    COL_TIME = 'key_year'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        


    def fit(self, X:pd.DataFrame, y=None, **kwargs) -> Self:
        self.data = X.copy()
        return super().fit(X, y, **kwargs)

    def transform(self, X:pd.DataFrame, **kwargs) -> ArrayLike:
        data = X.copy()
        data_extended = data.groupby(self.COL_GROUP).apply(self._extend).reset_index() 
        data_transformed = data_extended.groupby(self.COL_GROUP).progress_apply(self._interpolate).reset_index()
        return data_transformed

    def _trim(self, df_company: pd.DataFrame):
        df_company = df_company.sort_values(self.COL_TIME)
        years = df_company[self.COL_TIME].values
        df = df_company.set_index(self.COL_TIME)
        out = max(np.split(years, np.where(np.diff(years) != 1)[0] + 1), key=len).tolist()
        longest_sequence = df.loc[out]
        df_result = longest_sequence.reset_index()
        return df_result


    def _extend(self, df_company: pd.DataFrame):
        key_isin = df_company[self.COL_GROUP].values[0]
        min_year = df_company[self.COL_TIME].min()
        max_year = df_company[self.COL_TIME].max()
        rows = df_company.set_index(self.COL_TIME).to_dict()
        new_rows = [{i: rows.get(i, {self.COL_GROUP: key_isin})} for i in range(min_year, max_year+1)]
        new_data = pd.DataFrame(new_rows)
        return new_data

    def _interpolate(self, df_company: pd.DataFrame):
        results = df_company.interpolate()
        return results


    def evaluate(self, **kwargs) -> Self:
        data_trimmed = self.data.groupby(self.COL_GROUP).filter(lambda company: len(company) > 2)
        data_trimmed:pd.DataFrame = data_trimmed.groupby(self.COL_GROUP).progress_apply(self._trim)
        data_trimmed = data_trimmed.reset_index()
        super().evaluate(data_trimmed)
        return self

    def __repr__(self):
        return f"@{self.__class__.__name__}[ Count of imputed {self.imputed} ]"



