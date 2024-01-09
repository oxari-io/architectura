# %%
import os, sys
import pathlib

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

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_missing_value_imputers_lightgbm.csv', index_col=0)
df_results['specifier'] = df_results['imputer'] + '-' + df_results['lr'].astype(str)
df_results
# %%
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='difficulty', y='overall.sMAPE', hue='specifier')
ax.set_title('sMAPE over missingness difficulty for different learning rates')
plt.show()
# %%
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='lr', y='overall.sMAPE')
ax.set_title('sMAPE over learning rate')
ax.set_xscale('log')
plt.show()
# %%
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='n_estimators', y='overall.sMAPE')
ax.set_title('sMAPE over n_estimators')
plt.show()
# %%

df_matrix = df_results.pivot_table(index='lr', columns='n_estimators', values='overall.sMAPE', aggfunc='median')
df_matrix
# %%
plt.figure(figsize=(10, 6))
ax = sns.heatmap(data=df_matrix, cmap="viridis")
ax.set_title('Median sMAPE given the hyperparameters')
plt.show()

# %%
