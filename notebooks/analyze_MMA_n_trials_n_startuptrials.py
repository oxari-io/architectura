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
results = pd.read_csv(cwd.parent/'local/eval_results/experiment_MMA_n_trials_n_startup_trials.csv', index_col=0)
# %%
results
# %%
plt.figure(figsize=(10,10))
fig = sns.regplot(data=results, x="n_trials", y="test.sMAPE", order=3, color="blue")
ax2 = plt.twinx()
fig = sns.lineplot(data=results, x="n_trials", y="time", color="green", ax=ax2)
fig.set_title('two y axes: smape (blue) and time (green) vs n_trials')
# %%
plt.figure(figsize=(10,10))
fig = sns.regplot(data=results, x="n_startup_trials", y="test.sMAPE", order=3, color="blue")
ax2 = plt.twinx()
fig = sns.lineplot(data=results, x="n_startup_trials", y="time", color="green", ax=ax2)
fig.set_title('two y axes: smape (blue) and time (green) vs n_startup_trials')
# %%
np_n_trials = results["n_trials"].to_numpy()
np_n_startup_trials = results["n_startup_trials"].to_numpy()

#%%
# Binned heatmap: min sMAPE value for each bin
bins_x = np.arange(0, results['n_trials'].max() + 20, 20)
bins_y = np.arange(0, results['n_startup_trials'].max() + 2, 2)

results['n_trials_bin'] = pd.cut(results['n_trials'], bins=bins_x, right=False)
results['n_startup_trials_bin'] = pd.cut(results['n_startup_trials'], bins=bins_y, right=False)

pivot_table_sMAPE = results.pivot_table(index='n_startup_trials_bin', columns='n_trials_bin', values='test.sMAPE', aggfunc=np.min)
# pivot_table_time = results.pivot_table(index='n_startup_trials_bin', columns='n_trials_bin', values='time', aggfunc=np.mean)

plt.figure(figsize=(20, 10))
sns.heatmap(pivot_table_sMAPE, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Binned heatmap: n_trials vs n_startup_trials, heat is sMAPE')
plt.xlabel('n_trials')
plt.ylabel('n_startup_trials')
plt.show()

# plt.figure(figsize=(20, 10))
# sns.heatmap(pivot_table_time, annot=True, fmt=".3f", cmap="YlGnBu")
# plt.title('Binned heatmap: n_trials vs n_startup_trials, heat is time')
# plt.xlabel('n_trials')
# plt.ylabel('n_startup_trials')
# plt.show()

# %%
# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# hist, xedges, yedges = np.histogram2d(np_n_trials, np_n_startup_trials, bins=4, range=[[0, 4], [0, 4]])

# # Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0

# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# plt.show()
# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
results.rename(columns = {'test.sMAPE':'sMAPE'}, inplace = True)

#%%
# Results with lowest sMAPE
results_head = results.sort_values(by=['sMAPE']).head(10)
print(results_head[['n_trials', 'n_startup_trials', 'sMAPE', 'time']])

# %%
# results.head()
# %%
md = smf.mixedlm("sMAPE ~ n_trials + n_startup_trials", results, groups=results["data_split"])
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())
# %%