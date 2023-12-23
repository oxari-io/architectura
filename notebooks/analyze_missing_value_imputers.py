# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results_1 = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_run1.csv', index_col=0)
df_results_2 = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_run2.csv', index_col=0)
df_results_1["run"] = 1
df_results_2["run"] = 2

df_results = pd.concat([df_results_1, df_results_2]).sort_values('imputer')
# %%
# df_results["imputer"] = pd.Categorical(df_results["imputer"])
df_results["mae"] = df_results["overall.MAE"]
df_results["smape"] = df_results["overall.sMAPE"]
df_results = df_results[df_results["imputer"]!="DummyImputer"]
df_results
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="mae", y="imputer")
sns.boxplot(df_results, x="smape", y="imputer")
plt.show()
# %%
plt.figure(figsize=(15,15))
sns.lineplot(df_results, x="difficulty", y="smape", hue="imputer", errorbar=('ci', 0))
# plt.ylim(0,1)
plt.show()

# %%
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
sns.lineplot(df_results[df_results["name"].str.startswith("Revenue")], x="difficulty", y="smape", hue="name", errorbar=('se', 1))
plt.subplot(2,2,2)
sns.lineplot(df_results[df_results["name"].str.startswith("MVE")|df_results["name"].str.startswith("OldOxari")], x="difficulty", y="smape", hue="imputer", errorbar=('se', 1))
plt.subplot(2,2,3)
sns.lineplot(df_results[df_results["name"].str.startswith("Equi")], x="difficulty", y="smape", hue="imputer", errorbar=('se', 1))
plt.subplot(2,2,4)
sns.lineplot(df_results[df_results["name"].str.startswith("Categorical")], x="difficulty", y="smape", hue="imputer", errorbar=('se', 1))
plt.show()

# %%
plt.figure(figsize=(10,7))
condition = (df_results["name"].str.startswith("RevenueQuantile"))|(df_results["imputer"]=="MVEImputer:KNeighborsRegressor") |(df_results["imputer"]=="MVEImputer:LGBMRegressor")|(df_results["imputer"]=="OldOxariImputer:RandomForestRegressor")
sns.lineplot(df_results[condition], x="difficulty", y="smape", hue="imputer", errorbar=('se', 1))
plt.show()

# %%
formula = "mae ~ imputer + difficulty + mode"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["repetition"]).fit()
mod1.summary()
# %%
