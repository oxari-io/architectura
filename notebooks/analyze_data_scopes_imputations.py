# %%
import sys
sys.path.append("..")

from sklearn.preprocessing import PowerTransformer
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, FinancialLoader, ScopeLoader
from datasources.loaders import RegionLoader
from datasources.local import LocalDatasource
from imputers.revenue_bucket import RevenueQuantileBucketImputer

from postprocessors.missing_year_imputers import DerivativeMissingYearImputer
from preprocessors.core import IIDPreprocessor

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from base import OxariDataManager
from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
from datasources.online import S3Datasource
# sns.set_palette('viridis')
pd.set_option('display.float_format', lambda x: '%.5f' % x)
def get_company(df, key_isin):
    return df[df.key_isin==key_isin]
def keep_important_cols(df):
    return df.loc[:,["key_isin", "key_year", "ft_numc_revenue", "ft_numc_equity", "ft_numc_cash", "tg_numc_scope_1"]]


# %%
cwd = pathlib.Path(__file__).parent

# dataset = PreviousScopeFeaturesDataManager(
#         FinancialLoader(datasource=LocalDatasource(path=cwd.parent/"model-data/input/financials_auto.csv")),
#         ScopeLoader(datasource=LocalDatasource(path=cwd.parent/"model-data/input/scopes_auto.csv")),
#         CategoricalLoader(datasource=LocalDatasource(path=cwd.parent/"model-data/input/categoricals_auto.csv")),
#         RegionLoader(),
#     ).set_filter(CompanyDataFilter(1, True)).run()
# # %%
# DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
# bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
# SPLIT_1 = bag.scope_1
# SPLIT_2 = bag.scope_2
# SPLIT_3 = bag.scope_3

# company = list(set(DATA.key_isin))

# X,y = SPLIT_1.train
# X

# # %%
# preprocessor = IIDPreprocessor(fin_transformer=PowerTransformer()).set_imputer(RevenueQuantileBucketImputer(5)).fit(X, y)
# data_preprocessed = preprocessor.transform(DATA)
# data_preprocessed
# # %%
# my_imputer = DerivativeMissingYearImputer().fit(DATA)
# DATA_FOR_IMPUTE = my_imputer.transform(DATA)
# DATA_FOR_IMPUTE
# %%
data_scope_inputed = pd.read_csv(cwd.parent/'local/prod_runs/model_imputations_T202306040920.csv', index_col=0)
data_scope_inputed
 
# %%
def sample_groups(df, group_by_col, num_groups, filter_col=None):
    # group the dataframe by the specified column
    grouped = df.groupby(group_by_col)
    
    # get the unique group labels for groups with more than one row
    # and where not all rows in filter_col are True (if filter_col is not None)
    if filter_col is None:
        groups = [name for name, group in grouped if len(group) > 1]
    else:
        groups = [name for name, group in grouped if len(group) > 1 and not group[filter_col].all()]
    
    # if there are not enough groups with the specified characteristics
    if len(groups) < num_groups:
        print("Not enough groups with the specified characteristics.")
        return None
    
    # randomly select num_groups groups
    sampled_groups = np.random.choice(groups, num_groups, replace=False)
    
    # filter the original dataframe to include only the sampled groups
    sampled_df = df[df[group_by_col].isin(sampled_groups)]
    
    return sampled_df

col_to_normalize='tg_numc_scope_1'
df_scopes = sample_groups(data_scope_inputed, 'key_isin', 10, filter_col='predicted_s1')
df_scopes['normalized_scope_1'] = df_scopes.groupby('key_isin')[col_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_scopes

# %%
for idx, df_grp in df_scopes.groupby('key_isin'):
    ax = sns.lineplot(data=df_grp, y=col_to_normalize, x='key_year', color='grey')
    sns.scatterplot(data=df_grp, y=col_to_normalize, x='key_year', hue='predicted_s1')
    ax.set_title(idx)
    plt.show()
# %%
fig, ax= plt.subplots(1,1, figsize=(20,10))
sns.lineplot(data=df_scopes,y='normalized_scope_1', x='key_year', hue='key_isin', legend=False)
sns.scatterplot(data=df_scopes,y='normalized_scope_1', x='key_year', hue='key_isin', style='predicted_s1', s=100)
plt.show()

# %%
from IPython.display import display
# company = df_scopes.key_isin.unique()
def display_company(data_scope_inputed, col_to_normalize):
    fig, ax= plt.subplots(1,1, figsize=(13,8))
    sns.lineplot(data=data_scope_inputed,y=col_to_normalize, x='key_year', hue='key_isin', legend=False)
    sns.scatterplot(data=data_scope_inputed,y=col_to_normalize, x='key_year', hue='predicted_s1', s=100)
    display(data_scope_inputed.iloc[:, [2,0,3, 4, 6,7]])
    plt.show()

display_company(get_company(data_scope_inputed, 'US1720621010'), col_to_normalize)
# %%
display_company(get_company(data_scope_inputed, 'ZAE000003257'), col_to_normalize)
# %%
display_company(get_company(data_scope_inputed, 'ZAE000203238'), col_to_normalize)
# %%
display_company(get_company(data_scope_inputed, 'CH0012549785'), col_to_normalize)

# %%
