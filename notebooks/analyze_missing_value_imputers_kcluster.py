# %%
import os, sys
import pathlib

from sklearn.metrics import auc, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, FinancialLoader, ScopeLoader

from datasources.core import PreviousScopeFeaturesDataManager
from datasources.loaders import RegionLoader
from datasources.local import LocalDatasource

import pandas as pd
from base import OxariDataManager
from base.run_utils import get_small_datamanager_configuration
from sklearn.model_selection import train_test_split
import imputers.equilibrium_method as equi

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_palette('viridis')

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_missing_value_imputers_kcluster.csv', index_col=0)


# %%
fig = plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='difficulty', y='overall.sMAPE', hue='name', errorbar=None)
ax.set_title('sMAPE over missingness difficulty for different learning rates')
ax.axvline(x=0.3584905660377358, c='red', ls=':')
ax.text(x=0.3584905660377358+0.01, y=1.10, s="Median missiness rate of train data")
fig.tight_layout()
plt.show()
# %%
fig = plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='difficulty', y='overall.sMAPE', hue='imputer', errorbar=None)
ax.set_title('sMAPE over missingness difficulty for different learning rates')
ax.axvline(x=0.3584905660377358, c='red', ls=':')
ax.text(x=0.3584905660377358+0.01, y=1.10, s="Median missiness rate of train data")
fig.tight_layout()
plt.show()
# %%
fig = plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='difficulty', y='overall.sMAPE', hue='bucket_number', errorbar=None)
ax.set_title('sMAPE over missingness difficulty for different learning rates')
ax.axvline(x=0.3584905660377358, c='red', ls=':')
ax.text(x=0.3584905660377358+0.01, y=1.10, s="Median missiness rate of train data")
fig.tight_layout()
plt.show()




# %%
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='bucket_number', y='overall.sMAPE')
ax.set_title('sMAPE over learning rate')
# ax.set_xscale('log')
plt.show()

# %%

df_matrix = df_results.pivot_table(index=['name'], columns=['bucket_number'], values='overall.sMAPE', aggfunc='median')
df_matrix
# %%
plt.figure(figsize=(10, 6))
ax = sns.heatmap(data=df_matrix, cmap="viridis")
ax.set_title('Median sMAPE given the hyperparameters')
plt.show()




# %%
