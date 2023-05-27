# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_temporal_features.csv', index_col=0)
df_results["scope_estimator"] = pd.Categorical(df_results["scope_estimator"])
df_results["cols_used"] = df_results["cols_used"].str.replace('ft_numc_', '').fillna('')
df_results["uses_scope_1"] = df_results["cols_used"].str.contains('scope_1')
df_results["uses_scope_2"] = df_results["cols_used"].str.contains('scope_2')
df_results["uses_scope_3"] = df_results["cols_used"].str.contains('scope_3')
df_results["uses_year"] = df_results["cols_used"].str.contains('year')
df_results["clueless_impact"] = df_results["clueless.sMAPE"] - df_results["raw.sMAPE"]
df_results["clueless_impact_ex"] = df_results["clueless.sMAPE"] / df_results["raw.sMAPE"]

df_results
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

sns.boxplot(data=df_results, x="raw.sMAPE", y="cols_used", ax=ax1)
ax1.set_title('Normal')
sns.boxplot(data=df_results, x="clueless.sMAPE", y="cols_used", ax=ax2)
ax2.set_title('Without additional temporal data')
plt.show()
# %%
fig = plt.figure(figsize=(10, 10))
corr_matrix = df_results.filter(regex="^(raw\.sMAPE|raw\.R2|jump_rates\.)").corr()
sns.heatmap(corr_matrix, cmap="bwr", vmax=1, vmin=-1)

# %%
sns.boxplot(data=df_results, x="clueless_impact", y="cols_used")
plt.show()
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
sns.boxplot(data=df_results, x="jump_rates.normal.percentile50", y="cols_used", ax=ax1)
ax1.set_title('Normal')
sns.boxplot(data=df_results, x="jump_rates.clueless.percentile50", y="cols_used", ax=ax2)
ax2.set_title('Without additional temporal data')
plt.show()
# %%
fig = plt.figure(figsize=(10, 10))
sns.scatterplot(data=df_results, y="jump_rates.normal.percentile50", x="raw.sMAPE", hue="uses_year")
plt.show()
# %%
sns.boxplot(data=df_results, y="clueless_impact", x="uses_year")
plt.show()
# %%
sns.boxplot(data=df_results, y="clueless_impact", x="uses_scope_1")
plt.show()
# %%
sns.boxplot(data=df_results, y="clueless_impact", x="uses_scope_2")
plt.show()
# %%
sns.boxplot(data=df_results, y="clueless_impact", x="uses_scope_3")
plt.show()
# %%
