import numpy as np
import pandas as pd
from typing_extensions import Self

from pandas.core.computation.expressions import evaluate

from base import OxariMetaModel, OxariPostprocessor
import tqdm

from base.common import OxariEvaluator, OxariImputer, OxariLoggerMixin, OxariTransformer
from base.oxari_types import ArrayLike
import itertools as it

tqdm.tqdm.pandas()


class SimpleMissingYearImputer(OxariImputer):
    COL_GROUP = 'key_isin'
    COL_TIME = 'key_year'

    # class MYEvaluator(OxariImputer.DefaultImputerEvaluator):
    #     def evaluate(self, y_true, y_pred, **kwargs):
    #         return super().evaluate(y_true, y_pred, **kwargs)

    def __init__(self, method: str = 'linear', **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.data = X.copy().drop_duplicates([self.COL_GROUP, self.COL_TIME])
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        data = X.copy()
        data_extended = data.groupby(self.COL_GROUP, group_keys=False).apply(self._extend).reset_index(drop=True)
        data_transformed: pd.DataFrame = data_extended.infer_objects().groupby(self.COL_GROUP, group_keys=False).progress_apply(self._interpolate)
        data_transformed = data_transformed.reset_index(drop=True)
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
        min_year = int(df_company[self.COL_TIME].min())
        max_year = int(df_company[self.COL_TIME].max())
        rows = df_company.set_index(self.COL_TIME).to_dict(orient='index')
        new_rows = {i: {"key_year": i, "key_isin": key_isin, **rows.get(i, {})} for i in range(min_year, max_year + 1)}
        new_data = pd.DataFrame(new_rows).T
        return new_data

    def _interpolate(self, df_company: pd.DataFrame):
        results = df_company.interpolate(self.method)
        return results

    def _get_drop_indices(self, df_company: pd.DataFrame):
        allowed_indices:pd.Index = df_company.index[1:-2]
        drop_count = int(len(df_company) * self.difficulty)
        if len(allowed_indices) == 0:
            return []

        indices = np.random.choice(allowed_indices, min(drop_count, len(allowed_indices)), replace=False)
        return indices

    def evaluate(self, **kwargs) -> Self:
        self.difficulty = kwargs.get('difficulty', 0.1)
        data_trimmed = self.data.groupby(self.COL_GROUP).filter(lambda company: len(company) > 2)
        data_trimmed: pd.DataFrame = data_trimmed.groupby(self.COL_GROUP, group_keys=False).progress_apply(self._trim)
        data_trimmed = data_trimmed.reset_index(drop=True)

        X_true = data_trimmed.infer_objects()
        X_true_features = X_true.filter(regex="^ft_num", axis=1)
        ft_cols = X_true_features.columns

        mask = list(it.chain(*X_true.groupby(self.COL_GROUP, group_keys=False).apply(self._get_drop_indices).values))
        X_eval = X_true.copy()
        X_eval = X_eval.drop(mask)

        X_pred = self.transform(X_eval, **kwargs)

        y_true = np.array(X_true[ft_cols])[mask].flatten()
        y_pred = np.array(X_pred[ft_cols])[mask].flatten()
        self._evaluation_results = {}
        self._evaluation_results["overall"] = self._evaluator.evaluate(y_true, y_pred)

        return self

    def get_config(self, deep=True):
        return {"difficulty_level": self.difficulty, **super().get_config(deep)}

    def __repr__(self):
        return f"@{self.__class__.__name__}[ --- ]"


class CubicMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='cubicspline', **kwargs)

    def _interpolate(self, df_company: pd.DataFrame):
        results = df_company.interpolate(self.method)
        return results