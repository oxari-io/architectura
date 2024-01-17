# %%
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np


# %%
# See full results of the experiment
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_estimators_svm.csv', index_col=0)
results = results.replace([np.inf, -np.inf], 2, inplace=False)
# # %%
# grouped_data = results.groupby(['scope_estimator', 'scope'])['test.sMAPE'].mean().reset_index()

# plt.figure(figsize=(10, 6))
# fig = sns.barplot(
#     data=grouped_data, 
#     x='scope_estimator', 
#     y='test.sMAPE', 
#     hue='scope'
# )
# plt.title('test.sMAPE value for each estimator')
# plt.xlabel('Estimator')
# plt.ylabel('test.sMAPE Value')
# plt.legend(title='Scope')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()


# %%

plt.figure(figsize=(10, 6))
fig = sns.barplot(
    data=results, 
    x='scope_estimator', 
    y='raw.sMAPE', 
    # hue='scope'
)
plt.title('sMAPE value for each estimator')
plt.xlabel('Estimator')
plt.ylabel('sMAPE Value')
plt.legend(title='Scope')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# # %%
# closer_results = results[results['scope_estimator'].isin(['AdaboostEstimator', 'MiniModelArmyEstimator'])]

# grouped_data = closer_results.groupby(['scope_estimator', 'scope'])['test.sMAPE'].mean().reset_index()

# plt.figure(figsize=(10, 6))
# fig = sns.barplot(
#     data=grouped_data, 
#     x='scope_estimator', 
#     y='test.sMAPE', 
#     hue='scope'
# )
# plt.title('test.sMAPE value for each estimator')
# plt.xlabel('Estimator')
# plt.ylabel('test.sMAPE Value')
# plt.legend(title='Scope')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# %%
plt.figure(figsize=(10, 6))
fig = sns.boxplot(
    data=results, 
    x='scope_estimator', 
    y='raw.sMAPE'
)


plt.title('sMAPE value for each estimator - Scope 1')
plt.xlabel('Estimator')
plt.ylabel('sMAPE Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# %%
plt.figure(figsize=(10, 6))
fig = sns.boxplot(
    data=results, 
    x='scope_estimator', 
    y='stats.optimise.time'
)


plt.title('Time for each estimator - Scope 1')
plt.xlabel('Estimator')
plt.ylabel('Seconds')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# %%
