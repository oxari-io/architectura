# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent/'local/eval_results/experiment_PCA.csv', index_col=0)
# %%
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
plt.figure(figsize=(17,10))
fig = sns.regplot(data=results, x="n_components", y="test.sMAPE", order=3, color="blue")
ax2 = plt.twinx()
fig = sns.lineplot(data=results, x="n_components", y="time", color="green", ax=ax2)
fig.set_title('two y axes: smape (blue) and time (green) vs n_components')
# %%
plt.figure(figsize=(17,10))
fig = sns.regplot(data=results[results['time']<400], x="time", y="test.sMAPE")
fig.set_xlabel('time')
fig.set_ylabel('sMAPE')
fig.set_title('smape vs time (ROC plot)')
# %%
plt.figure(figsize=(17,10))
fig = sns.scatterplot(data=results, x="n_components", y="test.sMAPE", size="time")
fig.set_xlabel('n_components')
fig.set_ylabel('sMAPE')
fig.set_ylim(0.22, 0.28)
fig.set_title('bubble plot, size indicates time')
# %%
plt.figure(figsize=(17,10))
fig = sns.scatterplot(data=results, x="n_components", y="variance")
fig.set_xlabel('n_components')
fig.set_ylabel('variance')
fig.set_title('scree plot')

# %%
# idea 1: two y axes: smape and time vs n_components
# idea 2: smape vs time (ROC plot)
# bubble plot 