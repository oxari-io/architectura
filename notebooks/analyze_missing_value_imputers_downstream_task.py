# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_downstream_task.csv', index_col=0)
df_results

# %%
# df_results["imputer"] = pd.Categorical(df_results["imputer"])
df_results["mae"] = df_results["raw.MAE"]
df_results["smape"] = df_results["raw.sMAPE"]
df_results
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="mae", y="imputer")
sns.boxplot(df_results, x="smape", y="imputer")
plt.show()

# %%
