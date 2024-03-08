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
df_results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_missing_value_imputers_equilibrium.csv', index_col=0)
df_results['specifier'] = df_results['name'] + '-' + df_results['mims_tresh'].astype(str) + '-' + df_results['skip_cols'].astype(str)
df_results['convergence_thresh'] = df_results['mims_tresh']
df_results
# %%
fig = plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='difficulty', y='overall.sMAPE', hue='specifier', ci=None)
ax.axvline(x=0.3584905660377358, c='red', ls=':')
ax.text(x=0.3584905660377358+0.01, y=0.7, s="Median missiness rate of train data")
fig.tight_layout()
plt.show()
# %%
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_results, x='difficulty', y='statistics.transform_time', hue='convergence_thresh')
ax.set_title("Time to execute a transformation in seconds")
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_results, x='difficulty', y='completed_iter', hue='convergence_thresh')
ax.set_title("Number of completed iterations before early stop")
plt.show()


# %%
def vis_aspect_plot(col, df_results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(data=df_results, x='skip_cols', y=col, showfliers=True, ax=ax1)
    sns.boxplot(data=df_results, x='name', y=col, showfliers=True, ax=ax2)
    sns.boxplot(data=df_results, x='name', y=col, hue='skip_cols', ax=ax3)
    fig.suptitle(f'Effects of configuration on {col}')
    fig.tight_layout()
    plt.show()


vis_aspect_plot('overall.sMAPE', df_results[~df_results.name.str.startswith('Baseline')])
vis_aspect_plot('statistics.transform_time', df_results[~df_results.name.str.startswith('Baseline')])
vis_aspect_plot('completed_iter', df_results[~df_results.name.str.startswith('Baseline')])

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_results, x='skip_cols', y='overall.sMAPE', hue='name', showfliers=True)
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_results, x='skip_cols', y='statistics.transform_time', hue='name', showfliers=True)
plt.show()

# %%
plt.figure(figsize=(10, 6))
df_results['iter_groups'] = pd.cut(df_results['completed_iter'], bins=[0, 4, 15, 90,102])
sns.countplot(x='iter_groups', data=df_results, hue='skip_cols')
plt.show()

# %%
all_results = []
difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dataset = PreviousScopeFeaturesDataManager(
    FinancialLoader(datasource=LocalDatasource(path="../model-data/input/financials.csv")),
    ScopeLoader(datasource=LocalDatasource(path="../model-data/input/scopes.csv")),
    CategoricalLoader(datasource=LocalDatasource(path="../model-data/input/categoricals.csv")),
    RegionLoader(),
).set_filter(CompanyDataFilter(0.5)).run()

# %%
bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
SPLIT_1 = bag.scope_1
X, Y = SPLIT_1.train

# %%
X_new = X.copy()
X_train, X_test = train_test_split(X_new, test_size=0.5)
keeping_criterion_2 = (X_test.isna().mean(axis=0) < 0.3)
keep_columns_2 = X_train.loc[:, keeping_criterion_2].columns

# %%
import importlib

importlib.reload(equi)

# %%
imputer_1: equi.FastEquilibriumImputer = equi.FastEquilibriumImputer(verbose=False,
                                                                     max_iter=100,
                                                                     diff_tresh=0,
                                                                     mims_tresh=0.0001,
                                                                     max_diff_increase_thresh=0.75,
                                                                     skip_converged_cols=False).clone()
X_subset = X_train[keep_columns_2]
imputer_1 = imputer_1.fit(X_subset)
data = imputer_1.evaluate(X_test[keep_columns_2])
print(imputer_1.statistics)
imputer_1.visualize()
plt.show()

# %%
imputer_2: equi.EquilibriumImputer = equi.FastEquilibriumImputer(verbose=False,
                                                                 max_iter=100,
                                                                 diff_tresh=0,
                                                                 mims_tresh=0.0001,
                                                                 max_diff_increase_thresh=0.75,
                                                                 skip_converged_cols=True)

X_subset = X_train[keep_columns_2]
imputer_2 = imputer_2.fit(X_subset)
data = imputer_2.evaluate(X_test[keep_columns_2], )
print(imputer_2.statistics)
imputer_2.visualize()
plt.show()
# %%
imputer_3: equi.EquilibriumImputer = equi.EquilibriumImputer(verbose=False,
                                                             max_iter=100,
                                                                 diff_tresh=0,
                                                             mims_tresh=0.0001,
                                                             max_diff_increase_thresh=0.75,
                                                             skip_converged_cols=False)
X_subset = X_train[keep_columns_2]
imputer_3 = imputer_3.fit(X_subset)
data = imputer_3.evaluate(X_test[keep_columns_2])
print(imputer_3.statistics)
imputer_3.visualize()
plt.show()

# %%
imputer_4: equi.EquilibriumImputer = equi.EquilibriumImputer(verbose=False,
                                                             max_iter=100,
                                                             diff_tresh=0,
                                                             mims_tresh=0.0001,
                                                             max_diff_increase_thresh=0.6,
                                                             skip_converged_cols=True)
X_subset = X_train[keep_columns_2]
imputer_4 = imputer_4.fit(X_subset)
data = imputer_4.evaluate(X_test[keep_columns_2])
print(imputer_3.statistics)
imputer_4.visualize()
plt.show()

# %%
