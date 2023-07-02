# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent/'local/eval_results/experiment_MMA_n_trials_n_startup_trials_10_splits.csv', index_col=0)
# %%
plt.figure(figsize=(10,10))
fig = sns.regplot(data=results, x="n_trials", y="test.sMAPE", order=3, color="blue")
ax2 = plt.twinx()
fig = sns.lineplot(data=results, x="n_trials", y="time", color="green", ax=ax2)
fig.set_title('two y axes: smape (blue) and time (green) vs n_trials')
# %%
np_n_trials = results["n_trials"].to_numpy()
np_n_startup_trials = results["n_startup_trials"].to_numpy()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
hist, xedges, yedges = np.histogram2d(np_n_trials, np_n_startup_trials, bins=4, range=[[0, 4], [0, 4]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()
# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
results.rename(columns = {'test.sMAPE':'sMAPE'}, inplace = True)
# %%
results.head()
# %%
md = smf.mixedlm("sMAPE ~ n_trials + n_startup_trials", results, groups=results["data_split"])
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())
# %%
