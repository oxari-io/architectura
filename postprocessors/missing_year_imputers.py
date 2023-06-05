import numpy as np
import pandas as pd
from typing_extensions import Self

from pandas.core.computation.expressions import evaluate
from sklearn.preprocessing import MinMaxScaler

from base import OxariMetaModel, OxariPostprocessor
import tqdm

from base.common import OxariEvaluator, OxariImputer, OxariLoggerMixin, OxariTransformer
from base.oxari_types import ArrayLike
import itertools as it



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
        self.scaler = MinMaxScaler().fit(self.data.filter(regex="^ft_num", axis=1))
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        # data = X.filter(regex="^(ft_num|key_)", axis=1)
        data = X.copy()
        data_extended = data.groupby(self.COL_GROUP, group_keys=False).apply(self._extend).reset_index(drop=True)
        data_transformed: pd.DataFrame = data_extended.infer_objects().groupby(self.COL_GROUP, group_keys=False).progress_apply(self._interpolate).reset_index(drop=True)
        X_result = data_transformed.copy() 
        # Fill categorical vals
        data_completed = data_transformed.filter(regex=f'^(ft_cat|{self.COL_GROUP}|key_country)', axis=1).groupby(self.COL_GROUP, group_keys=False).apply(lambda x: x.bfill().ffill())
        X_result[data_completed.columns] = data_completed[data_completed.columns].values
        return X_result

    def _trim(self, df_company: pd.DataFrame):
        df_company = df_company.sort_values(self.COL_TIME)
        years = df_company[self.COL_TIME].values
        df = df_company.set_index(self.COL_TIME)
        out = max(np.split(years, np.where(np.diff(years) != 1)[0] + 1), key=len).tolist()
        longest_sequence = df.loc[out]
        df_result = longest_sequence.reset_index()
        return df_result

    def _extend(self, df_company: pd.DataFrame):

        df = df_company.set_index(self.COL_TIME).sort_index()
        key_isin = df_company[self.COL_GROUP].values[0]
        min_year = int(df.index.min())
        max_year = int(df.index.max())
        new_index = pd.RangeIndex(start=min_year, stop=max_year+1, name=self.COL_TIME)
        new_data = df.reindex(new_index).reset_index()
        new_data["key_isin"] = key_isin
        return new_data

    def _interpolate(self, df_company: pd.DataFrame):
        X_result = df_company.copy()
        results = df_company.filter(regex="^ft_num", axis=1).interpolate(self.method)
        X_result[results.columns] = results[results.columns].values
        return X_result

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

        
        X_true_matrix = X_true[ft_cols]
        X_pred_matrix = X_pred[ft_cols]
        
        X_true_matrix_scaled = self.scaler.transform(X_true_matrix)
        X_pred_matrix_scaled = self.scaler.transform(X_pred_matrix)

        y_true_scaled = np.array(X_true_matrix_scaled)[mask].flatten()
        y_pred_scaled = np.array(X_pred_matrix_scaled)[mask].flatten()

        y_true_raw = np.array(X_true_matrix)[mask].flatten()
        y_pred_raw = np.array(X_pred_matrix)[mask].flatten()

        row_mae = np.abs(np.array(X_true_matrix_scaled) - np.array(X_pred_matrix_scaled)).mean(axis=1)
        adj_mae = row_mae.sum()/(row_mae!=0).sum()

        self._evaluation_results = {}
        self._evaluation_results["scaled"] = self._evaluator.evaluate(y_true_scaled, y_pred_scaled+np.finfo(float).eps)
        self._evaluation_results["raw"] = self._evaluator.evaluate(y_true_raw, y_pred_raw+np.finfo(float).eps)
        self._evaluation_results["adjusted"] = {"mae":adj_mae}

        return self

    def get_config(self, deep=True):
        return {"difficulty_level": self.difficulty, **super().get_config(deep)}

    def __repr__(self):
        return f"@{self.__class__.__name__}[ --- ]"


class CubicSplineMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='cubicspline', **kwargs)


class QuadraticPolynomialMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='quadratic', **kwargs)


class DerivativeMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='from_derivatives', **kwargs)

class DummyMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='zero', **kwargs)

