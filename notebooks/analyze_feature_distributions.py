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

