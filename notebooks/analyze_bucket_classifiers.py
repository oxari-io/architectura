# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results_first = pd.read_csv(cwd.parent/'local/eval_results/experiment_bucket_classifier.csv', index_col=0)
# df_results_second = pd.read_csv(cwd.parent/'local/eval_results/experiment_bucket_optimum_continue.csv', index_col=0)[1:]
# df_results = pd.concat([df_results_first, df_results_second])
df_results = df_results_first
df_results["mae"] = df_results["raw.MAE"]
df_results["smape"] = df_results["raw.sMAPE"]

# %%
# %%
# sns.lineplot(data=df_results, x="test.n_buckets", y="smape")
ax = sns.boxplot(data=df_results, x="configuration", y="time")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
plt.show()
# %%
fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, sharex=True)
sns.boxplot(data=df_results, x="configuration", y="test.classifier.balanced_f1", ax=ax1)
sns.boxplot(data=df_results, x="configuration", y="train.classifier.balanced_f1", ax=ax2)
ax1.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax2.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()
# %%
ax = sns.boxplot(data=df_results, x="configuration", y="smape")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
plt.show()
# %%
ax = sns.scatterplot(data=df_results, x="test.classifier.balanced_f1", y="smape", hue="configuration")
plt.show()

 # %%
