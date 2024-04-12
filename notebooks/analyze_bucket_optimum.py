# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results_first = pd.read_csv(cwd.parent/'local/eval_results/experiment_bucket_optimum.csv', index_col=0)[1:]
# df_results_second = pd.read_csv(cwd.parent/'local/eval_results/experiment_bucket_optimum_continue.csv', index_col=0)[1:]
# df_results = pd.concat([df_results_first, df_results_second])
df_results = df_results_first
df_results["mae"] = df_results["raw.MAE"]
df_results["smape"] = df_results["raw.sMAPE"]

# %%
ax = sns.lineplot(data=df_results, x="test.n_buckets", y="time")
ax.set_ylabel("time (blue)")
plt.show()
# %%
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))
sns.lineplot(data=df_results, x="test.n_buckets", y="smape", ax=ax1,  color="r")
sns.lineplot(data=df_results, x="test.n_buckets", y="smape", ax=ax2,  color="r")
ax2.set_ylim(0,1)
fig.tight_layout()
plt.show()
# %%
ax2 = plt.gca()
sns.lineplot(data=df_results, x="test.n_buckets", y="test.sMAPE", ax=ax2,  color="r")
sns.lineplot(data=df_results, x="test.n_buckets", y="train.sMAPE", ax=ax2,  color="g")
ax2.set_ylabel("sMAPE (red)")
ax2.set_ylim(0,1)
plt.show()
# %%
