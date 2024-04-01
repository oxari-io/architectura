# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_linktransformer.csv', index_col=0)[1:]
df_results["mae"] = df_results["raw.MAE"]
df_results["smape"] = df_results["raw.sMAPE"]
df_results
# %%
# sns.lineplot(data=df_results, x="test.n_buckets", y="smape")
plt.figure(figsize=(10,5))
sns.boxplot(data=df_results, x="configuration", y="time")
plt.show()
# %%
plt.figure(figsize=(10,5))
sns.boxplot(data=df_results, x="configuration", y="smape")
plt.show()
# %%
df_results.groupby(["configuration"])[["smape", "time"]].describe().style.highlight_min(color = 'blue',  axis = 0).highlight_max(color = 'darkred',  axis = 0)
# %%
