# %%
import sys
sys.path.append("..")
from base.dataset_loader import CategoricalLoader, FinancialLoader, ScopeLoader
from datasources.loaders import RegionLoader

from datasources.local import LocalDatasource

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

sns.set_palette('viridis')

# %%
cwd = pathlib.Path(__file__).parent
DATA = pd.read_csv(cwd.parent/'model-data/input/scopes_auto.csv', index_col=0)
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
df_scopes['grp_scope_1'].value_counts()
# %%
sns.histplot(data=df_scopes[df_scopes["tg_numc_scope_1"] > 0], x="tg_numc_scope_1", bins=100)
# %%
# sns.histplot(data=df_scopes[(df_scopes["tg_numc_scope_1"] > 0) & (df_scopes["tg_numc_scope_1"] < 1e4)], x="tg_numc_scope_1", bins=100)
# %%
sns.histplot(data=df_scopes[df_scopes["tg_numc_scope_1"] > 0], x="tg_numc_scope_1", bins=100, log_scale=True)
# %%
df_scopes[df_scopes["grp_scope_1"] != "Zero Emissions"].groupby('key_year').var()
# %%
df_scopes[indices].groupby("key_year")

# %%
# sns.set_palette('viridis')
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
sns.kdeplot(data=df_scopes[df_scopes["tg_numc_scope_1"] > 0], x='tg_numc_scope_1', hue="key_year", log_scale=False, ax=ax, palette='viridis')
ax = fig.add_subplot(1, 2, 2)
sns.kdeplot(data=df_scopes[df_scopes["tg_numc_scope_1"] > 0], x='tg_numc_scope_1', hue="key_year", log_scale=True, ax=ax, palette='viridis')
fig.tight_layout()
plt.show()
# %%
# sns.kdeplot(data=df_scopes.groupby(['isin']).mean(), x="scope_1")
num_bins = 14
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

hist, xedges, yedges = np.histogram2d(df_scopes[indices]["key_year"], df_scopes[indices]["log_scope_1"], bins=num_bins)

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")

ax.plot_surface(xpos, ypos, hist, cmap='viridis', edgecolor='none')
ax.view_init(30, 220)
ax.set_xlabel('Year')
ax.set_ylabel('Scope 1 Emissions (log-scaled)')
ax.set_zlabel('Count')

fig.tight_layout()
plt.show()
# %%
num_bins = 14
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

hist, xedges, yedges = np.histogram2d(df_scopes[indices]["key_year"], df_scopes[indices]["log_scope_1"], bins=num_bins)

for row, year in zip(hist, xedges):
    xs = [year] * num_bins
    ys = row
    zs = yedges[:-1]
    ax.plot(xs, ys, zs, zdir="y")
    # ax.bar(xs, ys, zs, zdir="y")

ax.set_xlabel('Year')
ax.set_ylabel('Scope 1 Emissions (log-scaled)')
ax.set_zlabel('Count')

# On the y axis let's only label the discrete values that we have data for.
ax.set_yticks(np.round(yedges))
ax.invert_xaxis()
# ax.set_xticks(np.round(xedges))
ax.view_init(20, 25)
fig.tight_layout()
plt.show()

# %%
