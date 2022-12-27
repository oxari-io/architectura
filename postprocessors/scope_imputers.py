from base import OxariPostprocessor, OxariScopeEstimator, OxariDataManager, OxariMetaModel
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Union
from typing_extensions import Self

class ScopeImputerPostprocessor(OxariPostprocessor):
    def __init__(self, estimator: OxariMetaModel, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.jump_rate_evaluator = JumpRateEvaluator(self.estimator)
        self.imputed = {"scope_1": "N/A", "scope_2": "N/A", "scope_3": "N/A"}

    def run(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        # we are only interested in the most recent years
        data = X.loc[X["year"].isin(list(range(2016, 2022)))].copy()
        predicted_scope_1 = self.estimator.predict(data, scope=1)
        predicted_scope_2 = self.estimator.predict(data, scope=2)
        predicted_scope_3 = self.estimator.predict(data, scope=3)

        # adding a column that indicates whether the scope has been predicted or was reported
        data = data.assign(predicted_s1=np.where(data['scope_1'].isnull(), True, False))
        data = data.assign(predicted_s2=np.where(data['scope_2'].isnull(), True, False)) 
        data = data.assign(predicted_s3=np.where(data['scope_3'].isnull(), True, False))
        self.imputed = {f"scope_{k.split('_')[1][1]}": v for k, v in dict((data[['predicted_s1', 'predicted_s2', 'predicted_s3']] == True).sum()).items()}

        # filling missing values of scopes with model predictions
        data["scope_1"] = np.where(data['scope_1'].isnull(), predicted_scope_1, data['scope_1'])
        data["scope_2"] = np.where(data['scope_2'].isnull(), predicted_scope_2, data['scope_2'])
        data["scope_3"] = np.where(data['scope_3'].isnull(), predicted_scope_3, data['scope_3'])
        # TODO: Include logging how many predicted values where imputed.

        # retrieving only the relevant columns
        data = data[["isin", "year", "scope_1", "scope_2", "scope_3", "predicted_s1", "predicted_s2", "predicted_s3"]]
        # how many unique companies?
        print("Number of unique companies in the data: ", len(data["isin"].unique()))
        self.data = data
        return self

    def evaluate(self, **kwargs) -> Self:
        self.jump_rate_evaluator.evaluate(self.data, **kwargs)
        self.jump_rates = self.jump_rate_evaluator.jump_rates
        self.jump_rates_agg = self.jump_rate_evaluator.jump_rates_agg
        return self


    def __repr__(self):
        return f"@{self.__class__.__name__}[ Count of imputed {self.imputed} ]"

class JumpRateEvaluator():
    def __init__(self, estimator:OxariMetaModel) -> None:
        self.estimator = estimator 
        self.metrics = []
    
    def _compute_jump_rates(self, df_company:pd.DataFrame):
        columns = ['year', 'scope_1', 'scope_2', 'scope_3']
        
        pre, post = df_company.iloc[:-1][columns], df_company.iloc[1:][columns]
        jump_rate = post.values/pre
        jump_rate["year"] = pre['year'].astype('str').values + '-' + post['year'].astype('str').values
        
        # jump_rate_median = jump_rate.median(axis=0)
        return jump_rate
    
    def evaluate(self, X:pd.DataFrame) -> Self:
        companies = X.groupby('isin', group_keys=True)
        self.jump_rates = companies.apply(self._compute_jump_rates).reset_index().drop('level_1', axis=1)
        self.jump_rates_agg = self.jump_rates.drop('year', axis=1).groupby('isin').agg(['median', 'mean', 'std', 'max', 'min'])
        self.jump_rates_agg.columns = self.jump_rates_agg.columns.map('|'.join).str.strip('|')
        return self