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

    def __init__(self, method: str = 'linear', minimum_threshold=1, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.min_thresh = minimum_threshold

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        self.data = X.copy().drop_duplicates([self.COL_GROUP, self.COL_TIME])
        self.scaler = MinMaxScaler().fit(self.data.filter(regex="^ft_num", axis=1))
        return super().fit(X, y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> ArrayLike:
        # data = X.filter(regex="^(ft_num|key_)", axis=1)
        data = X.copy()
        self.logger.info('Extending data frame by missing years')
        data_extended: pd.DataFrame = data.groupby(self.COL_GROUP, group_keys=False).progress_apply(self._extend).reset_index(drop=True)
        self.logger.info('Filling data of the missing years')
        data_transformed: pd.DataFrame = data_extended.infer_objects().groupby(self.COL_GROUP, group_keys=False).progress_apply(self._interpolate).reset_index(drop=True)
        X_result = data_transformed.copy()
        # Fill categorical vals
        filter_str = f'^(ft_cat|{self.COL_GROUP}|key_country)'

        self.logger.info('Ffill and bfill categorical, meta and key fields')
        data_completed = data_transformed.filter(regex=filter_str, axis=1).groupby(self.COL_GROUP, group_keys=False).progress_apply(lambda x: x.bfill().ffill())
        X_result[data_completed.columns] = data_completed[data_completed.columns].values
        self.logger.info('Mark rows that where added during this process')
        X_result_marked = self.__mark_additional_rows(X_result, data, ['key_isin', 'key_year'], 'meta_is_imputed_year')
        self.logger.info('Done iwth scope imputation')
        return X_result_marked

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
        new_index = pd.RangeIndex(start=min_year, stop=max_year + 1, name=self.COL_TIME)
        new_data = df.reindex(new_index).reset_index()
        new_data["key_isin"] = key_isin
        return new_data

    def _interpolate(self, df_company: pd.DataFrame):
        X_result = df_company.copy()
        df = self.__extract_fillable_cols(df_company, self.min_thresh)
        results = df.interpolate(self.method)
        X_result[results.columns] = results[results.columns].values
        return X_result

    def _get_drop_indices(self, df_company: pd.DataFrame):
        allowed_indices: pd.Index = df_company.index[1:-2]
        drop_count = int(len(df_company) * self.difficulty)
        if len(allowed_indices) == 0:
            return []

        indices = np.random.choice(allowed_indices, min(drop_count, len(allowed_indices)), replace=False)
        return indices

    def __extract_fillable_cols(self, df_company: pd.DataFrame, min_points=1) -> pd.DataFrame:
        df = df_company.filter(regex="^ft_num", axis=1)
        elligible_cols = (df.count(axis=0) >= min_points).values
        filtered_data: pd.DataFrame = df.iloc[:, elligible_cols]
        results: pd.DataFrame = filtered_data.interpolate(self.method)
        return results

    def __mark_additional_rows(self, larger_df: pd.DataFrame, smaller_df: pd.DataFrame, columns: list, new_col: str) -> pd.DataFrame:
        """
        Marks rows in larger_df that are not present in smaller_df based on specified columns.

        Args:
        larger_df (pd.DataFrame): The larger dataframe.
        smaller_df (pd.DataFrame): The smaller dataframe.
        columns (list of str): List of column names to base the identification on.
        new_col (str): Name of the new column to be created in larger_df.

        Returns:
        pd.DataFrame: The larger dataframe with an additional column marking the additional rows.
        """

        # # create a key in both dataframes based on the specified columns
        # larger_df['key'] = larger_df[columns].astype(str).apply('_'.join, axis=1)
        # smaller_df['key'] = smaller_df[columns].astype(str).apply('_'.join, axis=1)

        # # create the new column in larger_df,
        # # it is True where 'key' of larger_df is not in 'key' of smaller_df, else False
        # larger_df[new_col] = ~larger_df['key'].isin(smaller_df['key'])

        # # drop the 'key' column in both dataframes as it's no longer needed
        # larger_df.drop('key', axis=1, inplace=True)
        # smaller_df.drop('key', axis=1, inplace=True)

        # return larger_df

        merged_df = pd.merge(larger_df, smaller_df[columns], on=columns, how='left', indicator=True)
        larger_df[new_col] = merged_df["_merge"] == "left_only"
        return larger_df

    def evaluate(self, **kwargs) -> Self:
        self.difficulty = kwargs.get('difficulty', 0.1)
        data_trimmed = self.data.groupby(self.COL_GROUP).filter(lambda company: len(company) > 2)
        data_trimmed: pd.DataFrame = data_trimmed.groupby(self.COL_GROUP, group_keys=False).progress_apply(self._trim)
        data_trimmed = data_trimmed.reset_index(drop=True)

        X_eval = data_trimmed.infer_objects()
        X_true = X_eval.copy()

        X_true_features = X_true.filter(regex="^ft_num", axis=1)
        ft_cols = X_true_features.columns

        mask = list(it.chain(*X_true.groupby(self.COL_GROUP, group_keys=False).apply(self._get_drop_indices).values))
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
        adj_mae = row_mae.sum() / (row_mae != 0).sum()

        self._evaluation_results = {}
        self._evaluation_results["scaled"] = self._evaluator.evaluate(y_true_scaled, y_pred_scaled + np.finfo(float).eps)
        self._evaluation_results["raw"] = self._evaluator.evaluate(y_true_raw, y_pred_raw + np.finfo(float).eps)
        self._evaluation_results["adjusted"] = {"mae": adj_mae}

        return self

    def get_config(self, deep=True):
        return {"difficulty_level": self.difficulty, **super().get_config(deep)}

    def __repr__(self):
        return f"@{self.__class__.__name__}[ --- ]"


class CubicSplineMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='cubicspline', minimum_threshold=4, **kwargs)


class QuadraticPolynomialMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='quadratic', minimum_threshold=3, **kwargs)


class DerivativeMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='from_derivatives', **kwargs)


class DummyMissingYearImputer(SimpleMissingYearImputer):

    def __init__(self, **kwargs):
        super().__init__(method='zero', **kwargs)
