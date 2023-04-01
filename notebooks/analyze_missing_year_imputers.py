# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_year_imputers.csv', index_col=0)
df_results
# %%
df_results["imputer"] = pd.Categorical(df_results["imputer"])
df_results["mae"] = df_results["adjusted.mae"]
df_results["smape"] = df_results["raw.sMAPE"]
df_results
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="scope_estimator", y="raw.sMAPE")
sns.boxplot(df_results, x="mae", y="imputer")
plt.show()
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="scope_estimator", y="raw.sMAPE")
sns.boxplot(df_results, x="imputer", y="mae", hue="difficulty")
# %%
plt.figure(figsize=(10,10))
sns.lineplot(df_results, x="difficulty", y="mae", hue="imputer")
plt.show()

# %%
formula = "smape ~ imputer + repetition"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["difficulty"]).fit()
mod1.summary()
# %%
