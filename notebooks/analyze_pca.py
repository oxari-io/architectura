# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# %%
cwd = pathlib.Path(__file__).parent
results = pd.concat([
    pd.read_csv(cwd.parent / 'local/eval_results/experiment_PCA_run_5.csv', index_col=0).assign(run=5),
    # pd.read_csv(cwd.parent / 'local/eval_results/experiment_PCA_run_1.csv', index_col=0).assign(run=1),
    # pd.read_csv(cwd.parent / 'local/eval_results/experiment_PCA_run_2.csv', index_col=0).assign(run=2),
    # pd.read_csv(cwd.parent / 'local/eval_results/experiment_PCA_run_3.csv', index_col=0).assign(run=3),
    pd.read_csv(cwd.parent / 'local/eval_results/experiment_PCA_run_4.csv', index_col=0).assign(run=4),
])
results
# %%
av_sMAPE = []
for i in range(1, 18):
    df_comp = results[results['n_components'] == i]
    av_sMAPE.append(df_comp['test.sMAPE'].median())
# %%
n_comps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# %%
# plt.figure(figsize=(17,10))
# fig = sns.regplot(x=n_comps, y=av_sMAPE, order=3)
# fig.set_xlabel('n_components')
# fig.set_ylabel('sMAPE')
# fig.set_title('median sMAPE over 10 runs vs n_components')
# # %%
# plt.figure(figsize=(17,10))
# fig = sns.regplot(data=results, x="n_components", y="test.sMAPE", order=3)
# fig.set_xlabel('n_components')
# fig.set_ylabel('sMAPE')
# fig.set_ylim(0.22, 0.28)
# fig.set_title('sMAPE limited between 0.22 and 0.28 vs n_components')
# # %%
# plt.figure(figsize=(17,10))
# fig = sns.scatterplot(data=results, x="time", y="test.sMAPE", hue="n_components")
# fig.set_xlabel('time')
# fig.set_ylabel('sMAPE')
# fig.set_ylim(0.22, 0.28)
# fig.set_title('sMAPE vs time, color indicates n_components')
# %%
plt.figure(figsize=(17, 10))
fig = sns.regplot(data=results, x="n_components", y="raw.sMAPE", order=3, color="blue")
# plt.ylim((0,1))
ax2 = plt.twinx()
fig = sns.regplot(data=results, x="n_components", y="time", color="green", order=3, ax=ax2)
fig.set_title('two y axes: smape (blue) and time (green) vs n_components')
plt.show()
# %%
plt.figure(figsize=(17, 10))
fig = sns.regplot(data=results, x="time", y="raw.sMAPE")
fig.set_xlabel('time')
fig.set_ylabel('sMAPE')
fig.set_title('smape vs time (ROC plot)')
plt.show()
# %%
plt.figure(figsize=(17, 10))
fig = sns.scatterplot(data=results, x="n_components", y="raw.sMAPE", size="time", hue="run")
fig.set_xlabel('n_components')
fig.set_ylabel('sMAPE')
# fig.set_ylim(0.22, 0.28)
fig.set_title('bubble plot, size indicates time')
plt.show()
# %%
plt.figure(figsize=(17, 10))
fig = sns.scatterplot(data=results, x="n_components", y="variance")
fig.set_xlabel('n_components')
fig.set_ylabel('variance')
fig.set_title('scree plot')
plt.show()

# %%
# idea 1: two y axes: smape and time vs n_components
# idea 2: smape vs time (ROC plot)
# bubble plot

#%%
plt.figure(figsize=(17, 10))
fig = sns.regplot(data=results, x='n_components', y='time', order=3)
fig.set_xlabel('n_components')
fig.set_ylabel('time')
fig.set_title('time vs n_components')
plt.show()
# %%
bins_x = np.arange(0, results['n_components'].max() + 10, 10)
bins_y = np.arange(0, results['time'].max() + 100, 100)
results['n_components_bin'] = pd.cut(results['n_components'], bins=bins_x, right=False)
results['time_bin'] = pd.cut(results['time'], bins=bins_y, right=False)
plt.figure(figsize=(17, 10))
fig = sns.regplot(data=results.groupby('n_components_bin').mean(), x='n_components', y='time', order=3)
fig.set_xlabel('n_components')
fig.set_ylabel('time')
fig.set_title('time vs n_components')
plt.show()
# %%
plt.figure(figsize=(17, 10))
fig = sns.boxplot(data=results, x='n_components_bin', y='time')
fig.set_xlabel('n_components')
fig.set_ylabel('time')
fig.set_title('time vs n_components')
plt.show()
# %%
print(len(results))

# %%
plt.figure(figsize=(17, 10))
fig = sns.heatmap(results.pivot_table(index='n_components_bin', columns='time_bin', values='raw.sMAPE', aggfunc=np.median), annot=True, fmt=".3f", cmap="YlGnBu")
ax = plt.gca()
ax.invert_yaxis()
fig.set_xlabel('time')
fig.set_ylabel('n_components')
fig.set_title('time vs n_components')
plt.show()
# %%
