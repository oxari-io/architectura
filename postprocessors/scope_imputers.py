import numpy as np
import pandas as pd
from typing_extensions import Self

from base import OxariMetaModel, OxariPostprocessor
import tqdm

from base.common import OxariLoggerMixin
tqdm.tqdm.pandas()

class ScopeImputerPostprocessor(OxariPostprocessor):

    def __init__(self, estimator: OxariMetaModel, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.jump_rate_evaluator = JumpRateEvaluator(self.estimator)
        self.imputed = {"tg_numc_scope_1": "N/A", "tg_numc_scope_2": "N/A", "tg_numc_scope_3": "N/A"}

    def run(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # we are only interested in the most recent years
        data = X.loc[X["key_year"].isin(list(range(2016, 2022)))].copy()
        predicted_scope_1 = self.estimator.predict(data, scope=1)
        predicted_scope_2 = self.estimator.predict(data, scope=2)
        predicted_scope_3 = self.estimator.predict(data, scope=3)

        # adding a column that indicates whether the scope has been predicted or was reported
        data = data.assign(predicted_s1=np.where(data['tg_numc_scope_1'].isnull(), True, False))
        data = data.assign(predicted_s2=np.where(data['tg_numc_scope_2'].isnull(), True, False))
        data = data.assign(predicted_s3=np.where(data['tg_numc_scope_3'].isnull(), True, False))
        self.imputed = {f"tg_numc_scope_{k.split('_')[1][1]}": v for k, v in dict((data[['predicted_s1', 'predicted_s2', 'predicted_s3']] == True).sum()).items()}

        # filling missing values of scopes with model predictions
        data["tg_numc_scope_1"] = np.where(data['tg_numc_scope_1'].isnull(), predicted_scope_1, data['tg_numc_scope_1'])
        data["tg_numc_scope_2"] = np.where(data['tg_numc_scope_2'].isnull(), predicted_scope_2, data['tg_numc_scope_2'])
        data["tg_numc_scope_3"] = np.where(data['tg_numc_scope_3'].isnull(), predicted_scope_3, data['tg_numc_scope_3'])
        # TODO: Include logging how many predicted values where imputed.

        # retrieving only the relevant columns
        data = data[["key_isin", "key_year", "tg_numc_scope_1", "tg_numc_scope_2", "tg_numc_scope_3", "predicted_s1", "predicted_s2", "predicted_s3"]]
        # how many unique companies?
        # print("Number of unique companies in the data: ", len(data["isin"].unique()))
        self.logger.debug(f"Number of unique companies in the data: {len(data['key_isin'].unique())}")
        self.data = data
        return self

    def evaluate(self, **kwargs) -> Self:
        self.jump_rate_evaluator.evaluate(self.data, **kwargs)
        self.jump_rates = self.jump_rate_evaluator.jump_rates
        self.jump_rates_agg = self.jump_rate_evaluator.jump_rates_agg
        return self

    def __repr__(self):
        return f"@{self.__class__.__name__}[ Count of imputed {self.imputed} ]"


class JumpRateEvaluator(OxariLoggerMixin):

    def __init__(self, estimator: OxariMetaModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.estimator = estimator
        self.metrics = []

    def _compute_jump_rates(self, df_company: pd.DataFrame):
        columns = ['key_year', 'tg_numc_scope_1', 'tg_numc_scope_2', 'tg_numc_scope_3']
        pre, post = df_company.iloc[:-1][columns], df_company.iloc[1:][columns]
        jump_rate = post.values / pre
        jump_rate["year_transition"] = pre['key_year'].astype('str').values + '-' + post['key_year'].astype('str').values

        return jump_rate.drop('key_year', axis=1)

    def _compute_estimate_to_fact_ratio(self, df_company: pd.DataFrame):
        columns_predicted = ["predicted_s1", "predicted_s2", "predicted_s3"]
        df_tmp = df_company[columns_predicted]
        num_datapoints = len(df_tmp)

        tmp_1 = {f"pred_num_s{idx+1}": val for idx, val in enumerate(df_tmp.sum(axis=0).to_dict().values())}
        tmp_2 = {f"pred_ratio_s{idx+1}": val for idx, val in enumerate((df_tmp.sum(axis=0) / num_datapoints).to_dict().values())}
        years = "-".join(df_company["key_year"].astype(str))
        result_series = pd.Series({**tmp_1, **tmp_2, "num_years": num_datapoints, "year_with_data": years})

        return result_series

    def evaluate(self, X: pd.DataFrame) -> Self:
        companies = X.groupby('key_isin', group_keys=True)
        self.logger.info("Compute Jump Rates")
        jump_rates: pd.DataFrame = companies.progress_apply(self._compute_jump_rates).reset_index().drop('level_1', axis=1).reset_index()
        self.logger.info("Compute Jump Ratios")
        estimation_stats: pd.DataFrame = companies.progress_apply(self._compute_estimate_to_fact_ratio).reset_index()
        self.jump_rates = jump_rates.merge(estimation_stats, left_on="key_isin", right_on="key_isin").drop('index', axis=1)
        # self.jump_rates = self.jump_rates.drop('level_1', axis=1)
        self.jump_rates_agg = self.jump_rates.drop('year_with_data', axis=1).groupby('key_isin').agg(['median', 'mean', 'std', 'max', 'min'])
        self.jump_rates_agg.columns = self.jump_rates_agg.columns.map('|'.join).str.strip('|')
        return self