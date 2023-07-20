# %%
import sys

sys.path.append("..")
import pathlib
from IPython.display import display

from datasources.loaders import RegionLoader
from datasources.local import LocalDatasource
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, FinancialLoader, ScopeLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from base import OxariDataManager
from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
from datasources.online import S3Datasource
from pathlib import Path

sns.set_palette('viridis')

PARENT_PATH = Path('..').absolute().resolve().as_posix()
PARENT_PATH
# %%
dataset = PreviousScopeFeaturesDataManager(
    FinancialLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/financials_auto.csv")),
    ScopeLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/scopes_auto.csv")),
    CategoricalLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/categoricals_auto.csv")),
    RegionLoader(),
).set_filter(CompanyDataFilter(frac=1)).run()
DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
DATA
# %%
df_scopes = DATA
df_scopes["grp_scope_1"] = None
df_scopes["log_scope_1"] = None
df_scopes.loc[df_scopes["tg_numc_scope_1"].isna(), ["grp_scope_1"]] = "Not reported"
df_scopes.loc[df_scopes["tg_numc_scope_1"] == 0, ["grp_scope_1"]] = "Zero Emissions"
df_scopes.loc[df_scopes["tg_numc_scope_1"] < 0, ["grp_scope_1"]] = "Impossible"
df_scopes.loc[df_scopes["tg_numc_scope_1"].between(0, 1, inclusive='right'), ["grp_scope_1"]] = "Weird"
df_scopes.loc[df_scopes["tg_numc_scope_1"] > 1, ["grp_scope_1"]] = "Emittor"
df_scopes["log_scope_1"] = np.log(df_scopes["tg_numc_scope_1"])
indices = df_scopes["tg_numc_scope_1"] > 0
df_scopes

# %%
numerical_features = df_scopes.filter(regex="^ft_numc", axis=1)
structure = pd.concat([numerical_features.skew(), numerical_features.kurtosis()], axis=1)
structure.columns = ['skew', 'kurstosis']
structure
# %%
# https://www.researchgate.net/publication/233967063_A_bi-symmetric_log_transformation_for_wide-range_data
# https://support.cytobank.org/hc/en-us/articles/206148057-About-the-Arcsinh-transform: The hyperbolic arcsine (arcsinh) is a function used in Cytobank for transforming data. It serves a similar purpose as transformation functions such as biexponential, logicle, hyperlog, etc.
# https://dillonhammill.github.io/CytoExploreR/articles/CytoExploreR-Transformations.html
# https://opendatascience.com/transforming-skewed-data-for-machine-learning/
#   Skewness test with shapiro-wilk test

num_bins = 30
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
faxes = axes.flatten()
for ax, feature in zip(faxes, numerical_features.columns):
    tmp_df = df_scopes
    # tmp_df = tmp_df.dropna(how="any", subset=[target, feature])

    # sns.scatterplot(tmp_df, x=feature, y=target, ax=ax)

    sns.histplot(x=np.arcsinh(tmp_df[feature]), ax=ax, bins=num_bins)
    # sns.histplot(x=tmp_df[feature], ax=ax, bins=num_bins)
fig.tight_layout()
plt.show()
# %%
num_bins = 14
fig, axes = plt.subplots(13, 4, figsize=(20, 60))
faxes = axes.flatten()

years = df_scopes["key_year"].unique()
target = "tg_numc_scope_1"
# target = "log_scope_1"
for axs, feature in zip(axes, numerical_features.columns):
    tmp_df = df_scopes[[target, feature]]
    tmp_df = tmp_df.dropna(how="any")

    ax1, ax2, ax3, ax4 = axs
    sns.scatterplot(x=np.arcsinh(tmp_df[feature]), y=tmp_df[target], ax=ax1, label="X-Scaled")
    sns.scatterplot(x=tmp_df[feature], y=np.arcsinh(tmp_df[target]), ax=ax2, label="Y-Scaled")
    sns.scatterplot(x=np.arcsinh(tmp_df[feature]), y=np.arcsinh(tmp_df[target]), ax=ax3, label="XY-Scaled")
    sns.kdeplot(x=np.arcsinh(tmp_df[feature]), y=np.arcsinh(tmp_df[target]), ax=ax4, label="XY-Scaled")

fig.tight_layout()
plt.show()

# %%
thresh = 0.5
from sklearn.impute import KNNImputer, SimpleImputer

# %%
correlations_original = numerical_features.corr()
correlations_original
# %%
plt.figure(figsize=(25, 20))
sns.heatmap(correlations_original.abs(), vmin=-1, vmax=1, cmap='bwr')

# %%
correlations = correlations_original.copy()
flag = True
while flag:
    highest_corrs = list(np.sum(correlations.abs() > thresh).sort_values().items())[-1]
    if highest_corrs[1] < 2:
        break
    # print(f"Going to remove {highest_corrs}")
    correlations = correlations.drop(highest_corrs[0], axis=1).drop(highest_corrs[0], axis=0)
    # display(correlations)
print('Iterative elimination\n')
print(f"features_iterative_corr_elimination = {list(correlations.columns)}")      
# %%
plt.figure(figsize=(25, 20))
sns.heatmap(correlations.abs(), vmin=-1, vmax=1, cmap='bwr')
# %%
correlations_strict = correlations_original.copy()
l_highest_corrs = list(np.sum(correlations_strict.abs() > thresh).sort_values(ascending=0).items())
reversed(l_highest_corrs)
for key, val in l_highest_corrs:
    if val > 1:
        # print(f"Going to remove {(key, val)}")
        correlations_strict = correlations_strict.drop(key, axis=1).drop(key, axis=0)

print('Strict elimination\n')
print(f"features_strict_corr_elimination = {list(correlations_strict.columns)}")  
# %%
plt.figure(figsize=(25, 20))
sns.heatmap(correlations_strict.abs(), vmin=-1, vmax=1, cmap='bwr')
# %%
numerical_features = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(numerical_features), columns=numerical_features.columns, index=numerical_features.index)
numerical_features
# %%
# https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

# define number of features to keep

k = 10

# perform feature selection
y = DATA['tg_numc_scope_1']
selector = SelectKBest(f_regression, k=k).fit(numerical_features[~y.isna()], y[~y.isna()])
X_new = selector.transform(numerical_features)
# get feature names of selected features

selected_features = numerical_features.columns[selector.get_support()]

# print selected features
print('SelectKBest elimination\n')
print(f"features_select_k_best = {list(selected_features)}")
# %%
plt.figure(figsize=(25, 20))
sns.heatmap(numerical_features[selected_features].corr().abs(), vmin=-1, vmax=1, cmap='bwr')

# %%
# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
# calculate VIF for each feature

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(numerical_features, i) for i in tqdm(range(numerical_features.shape[1]))]

vif["features"] = numerical_features.columns

# print VIF values
# %%
print('VIF elimination\n')
print(f"features_VIF_under_10 = {vif[vif['VIF Factor'] < 10].features.tolist()}")
# %%
plt.figure(figsize=(25, 20))
sns.heatmap(numerical_features[vif[vif["VIF Factor"] < 10].features.tolist()].corr().abs(), vmin=-1, vmax=1, cmap='bwr')
# %%
print('VIF elimination\n')
print(f"features_VIF_under_5 = {vif[vif['VIF Factor'] < 5].features.tolist()}")
# %%
plt.figure(figsize=(25, 20))
sns.heatmap(numerical_features[vif[vif["VIF Factor"] < 5].features.tolist()].corr().abs(), vmin=-1, vmax=1, cmap='bwr')
# %%
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

estimator = RandomForestRegressor()
selector = RFECV(estimator, step=0.1, cv=10, verbose=True)
selector = selector.fit(numerical_features[~y.isna()], y[~y.isna()])

# %%

plt.figure(figsize=(25, 20))
sns.heatmap(numerical_features.iloc[:, selector.support_].corr().abs(), vmin=-1, vmax=1, cmap='bwr')

# %%
