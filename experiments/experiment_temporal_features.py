# import joblib as pkl
# from dataset_loader.csv_loader import CSVScopeLoader, CSVFinancialLoader, CSVCategoricalLoader
import time

import pandas as pd
import numpy as np

from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator, DummyConfidenceEstimator, ConformalKNNConfidenceEstimator, JacknifeConfidenceEstimator
from base.common import DefaultRegressorEvaluator
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, FinancialLoader, ScopeLoader
from datasources.core import PreviousScopeFeaturesDataManager, TemporalFeaturesDataManager
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from datasources.loaders import RegionLoader
from datasources.local import LocalDatasource
from feature_reducers import PCAFeatureReducer
# from imputers.revenue_bucket import RevenueBucketImputer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import MiniModelArmyEstimator, SupportVectorEstimator
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
from base.helper import ArcSinhScaler, DummyFeatureScaler, DummyTargetScaler, LogTargetScaler
import tqdm
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def compute_jump_rates(df_company: pd.DataFrame):
    columns = ['key_year', 'tg_numc_scope_1']
    pre, post = df_company.iloc[:-1][columns], df_company.iloc[1:][columns]
    jump_rate = pd.DataFrame(np.maximum(post.values, pre.values)/np.minimum(post.values, pre.values), columns=["key_year", 'tg_numc_scope_1'])
    jump_rate["year_transition"] = pre['key_year'].astype('str').values + '-' + post['key_year'].astype('str').values
    return jump_rate.drop('key_year', axis=1)


def evaluate_jump_rates(ppl1, data):
    meta_is_pred_scope_1 = ppl1.predict(data)
    # adding a column that indicates whether the scope has been predicted or was reported
    data = data.assign(meta_is_pred_s1=np.where(data['tg_numc_scope_1'].isnull(), True, False))
    # filling missing values of scopes with model predictions
    data["tg_numc_scope_1"] = np.where(data['tg_numc_scope_1'].isnull(), meta_is_pred_scope_1, data['tg_numc_scope_1'])
    # retrieving only the relevant columns
    meta_keys = list(data.columns[data.columns.str.startswith('key_')])
    scope_keys = list(data.columns[data.columns.str.startswith('tg_numc_')])
    data = data[meta_keys + scope_keys + ["meta_is_pred_s1"]]

    companies = data.groupby('key_isin', group_keys=True)
    jump_rates: pd.DataFrame = companies.progress_apply(compute_jump_rates).reset_index().drop('level_1', axis=1).reset_index()
    jr_series = jump_rates["tg_numc_scope_1"].replace([np.inf, -np.inf], np.nan)
    jr_results = {
        "min": jr_series.min(),
        "percentile10": jr_series.quantile(0.1),
        "percentile25": jr_series.quantile(0.25),
        "percentile50": jr_series.quantile(0.5),
        "percentile75": jr_series.quantile(0.75),
        "percentile90": jr_series.quantile(0.9),
        "max": jr_series.max(),
        "std": jr_series.std(),
        "mean": jr_series.mean(),
    }

    return jr_results


if __name__ == "__main__":

    all_results = []
    num_repeats = 20
    # loads the data just like CSVDataLoader, but a selection of the data

    pbar = tqdm.tqdm(total=num_repeats * len(list(powerset([1,2,3,4]))), desc="Overall")

    for i in range(num_repeats):
        dataset = TemporalFeaturesDataManager(
            FinancialLoader(datasource=LocalDatasource(path="model-data/input/financials_auto.csv")),
            ScopeLoader(datasource=LocalDatasource(path="model-data/input/scopes_auto.csv")),
            CategoricalLoader(datasource=LocalDatasource(path="model-data/input/categoricals_auto.csv")),
            RegionLoader(),
        ).set_filter(CompanyDataFilter(0.25, drop_single_rows=True)).run()  # run() calls _transform()
        evaluator = DefaultRegressorEvaluator()

        bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
        DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
        columns = set(DATA.filter(regex=f"^{PreviousScopeFeaturesDataManager.PREFIX}", axis=1).columns)
        columns.add("ft_numd_year")
        all_combinations = list(powerset(columns))
        all_combinations.reverse()
        SPLIT_1 = bag.scope_1
        SPLIT_2 = bag.scope_2
        SPLIT_3 = bag.scope_3

        X, y = SPLIT_1.train

        for to_drop in all_combinations:

            # this loop runs a pipeline with each of the feature selection methods that were given as command line arguments, by default compare all methods
            results = []  # dictionary where key=feature selection method, value = evaluation results

            X_train, y_train = SPLIT_1.train
            X_val, y_val = SPLIT_1.val

            X_train_normal: pd.DataFrame = X_train.drop(list(to_drop), axis=1) if len(to_drop) else X_train
            X_val_normal: pd.DataFrame = X_val.drop(list(to_drop), axis=1) if len(to_drop) else X_val

            start = time.time()
            remaining_cols = "|".join(columns.difference(to_drop))

            ppl1 = DefaultPipeline(
                preprocessor=IIDPreprocessor(fin_transformer=PowerTransformer()),
                feature_reducer=PCAFeatureReducer(),
                imputer=RevenueQuantileBucketImputer(),
                scope_estimator=MiniModelArmyEstimator(),
                ci_estimator=None,
                scope_transformer=LogTargetScaler(),
            ).optimise(X_train_normal, y_train).fit(X_train_normal, y_train).evaluate(X_train_normal, y_train, X_val_normal, y_val)

            # CLUELESS PREDICTION
            X_val_clueless = X_val.copy().drop(list(columns), axis=1, errors='ignore')
            clueless_results = evaluator.evaluate(y_val.values, ppl1.predict(X_val_clueless))

            # JUMP RATE EVALUATION
            jr_results = evaluate_jump_rates(ppl1, DATA.copy())
            jr_results_clueless = evaluate_jump_rates(ppl1, DATA.copy().drop(list(columns), axis=1, errors='ignore'))
            all_results.append({
                "repetition": i + 1,
                "cols_used": remaining_cols,
                "time": time.time() - start,
                "scope": 1,
                "jump_rates": {
                    "normal": jr_results,
                    "clueless": jr_results_clueless,
                },
                "clueless": clueless_results,
                **ppl1.evaluation_results,
            })
            ### EVALUATION RESULTS ###
            concatenated = pd.json_normalize(all_results)
            fname = __loader__.name.split(".")[-1]
            concatenated.to_csv(f'local/eval_results/{fname}.csv')

            pbar.update(1)
