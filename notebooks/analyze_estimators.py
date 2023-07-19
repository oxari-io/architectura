# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# %%
# See full results of the experiment
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_estimators.csv', index_col=0)


# %%
grouped_data = results.groupby(['scope_estimator', 'scope'])['test.sMAPE'].mean().reset_index()

plt.figure(figsize=(10, 6))
fig = sns.barplot(
    data=grouped_data, 
    x='scope_estimator', 
    y='test.sMAPE', 
    hue='scope'
)
plt.title('test.sMAPE value for each estimator')
plt.xlabel('Estimator')
plt.ylabel('test.sMAPE Value')
plt.legend(title='Scope')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# %%
grouped_data = results.groupby(['scope_estimator', 'scope'])['raw.sMAPE'].mean().reset_index()

plt.figure(figsize=(10, 6))
fig = sns.barplot(
    data=grouped_data, 
    x='scope_estimator', 
    y='raw.sMAPE', 
    hue='scope'
)
plt.title('raw.sMAPE value for each estimator')
plt.xlabel('Estimator')
plt.ylabel('raw.sMAPE Value')
plt.legend(title='Scope')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# %%
closer_results = results[results['scope_estimator'].isin(['AdaboostEstimator', 'MiniModelArmyEstimator'])]

grouped_data = closer_results.groupby(['scope_estimator', 'scope'])['test.sMAPE'].mean().reset_index()

plt.figure(figsize=(10, 6))
fig = sns.barplot(
    data=grouped_data, 
    x='scope_estimator', 
    y='test.sMAPE', 
    hue='scope'
)
plt.title('test.sMAPE value for each estimator')
plt.xlabel('Estimator')
plt.ylabel('test.sMAPE Value')
plt.legend(title='Scope')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# %%
