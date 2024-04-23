# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects


# %%
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_model_leap_forward_1.csv', index_col=0)
results["configuration"] = results["c_imputer"] + "-" + results["c_model"] + "-" + results["c_fintransformer"] + "-" + results["c_preprocessor"]
results
# %%
plt.figure(figsize=(10,10))
ax = sns.boxplot(results, x="c_imputer", y="raw.sMAPE")
plt.xticks(rotation=60)
plt.show()
# %%
plt.figure(figsize=(10,10))
ax = sns.boxplot(results, x="c_model", y="raw.sMAPE")
plt.xticks(rotation=60)
plt.show()
# %%
plt.figure(figsize=(10,10))
ax = sns.boxplot(results, x="c_fintransformer", y="raw.sMAPE")
plt.xticks(rotation=60)
plt.show()
# %%
plt.figure(figsize=(10,10))
ax = sns.boxplot(results, x="c_preprocessor", y="raw.sMAPE")
plt.xticks(rotation=60)
plt.show()
 # %%
plt.figure(figsize=(10,10))
ax = sns.boxplot(results.sort_values("raw.sMAPE"), x="configuration", y="raw.sMAPE")
plt.xticks(rotation=90)
plt.show()
# %%
results.groupby("configuration")["raw.sMAPE"].describe().sort_values("50%").drop(columns=("count")).style.highlight_min(color = 'blue',  
                       axis = 0).highlight_max(color = 'darkred',  
                       axis = 0)
# %%
