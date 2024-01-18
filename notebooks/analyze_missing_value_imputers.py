# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_bkp.csv', index_col=0)

# df_results_1 = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_run1.csv', index_col=0)
# df_results_2 = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_run2.csv', index_col=0)
# df_results_1["run"] = 1
# df_results_2["run"] = 2

# df_results = pd.concat([df_results_1, df_results_2]).sort_values('imputer')
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
sns.lineplot(df_results[df_results["name"].str.contains("Quantile")], x="difficulty", y="smape", hue="name", errorbar=('se', 1))
plt.subplot(2,2,2)
sns.lineplot(df_results[df_results["name"].str.startswith("MVE")|df_results["name"].str.startswith("OldOxari")], x="difficulty", y="smape", hue="imputer", errorbar=('se', 1))
plt.subplot(2,2,3)
sns.lineplot(df_results[df_results["name"].str.startswith("K")], x="difficulty", y="smape", hue="imputer", errorbar=('se', 1))
plt.subplot(2,2,4)
sns.lineplot(df_results[df_results["name"].str.startswith("Categorical")], x="difficulty", y="smape", hue="imputer", errorbar=('se', 1))
plt.show()

# %%
plt.figure(figsize=(10,7))
condition = (df_results["name"].str.startswith("TotalAssetsQuantile"))|(df_results["name"].str.startswith("Equilibrium"))|(df_results["name"].str.startswith("KNN"))|(df_results["imputer"]=="MVEImputer:LGBMRegressor")|(df_results["imputer"]=="OldOxariImputer:RandomForestRegressor")|(df_results["imputer"].str.contains("country_code"))
ax = sns.lineplot(df_results[condition], x="difficulty", y="smape", hue="imputer", errorbar=('se', 1))
ax.set_title("Best of Breed comparison")
plt.show()
# %%
plt.figure(figsize=(10,7))
condition = (df_results["name"].str.startswith("TotalAssetsQuantile"))|(df_results["name"].str.startswith("Equilibrium"))|(df_results["name"].str.startswith("KNN"))|(df_results["imputer"]=="MVEImputer:LGBMRegressor")|(df_results["imputer"]=="OldOxariImputer:RandomForestRegressor")|(df_results["imputer"].str.contains("country_code"))
ax = sns.lineplot(df_results[condition], x="difficulty", y="smape",style="mode", hue="imputer", errorbar=('se', 1))
ax.set_title("Best of Breed comparison")
plt.show()
# %%
plt.figure(figsize=(10,7))
ax = sns.lineplot(df_results, x="difficulty", y="smape",hue="mode", errorbar=('se', 1))
ax.set_title("Best of Breed comparison")
plt.show()

# %%
formula = "mae ~ imputer + difficulty + mode"
mod1 = smf.mixedlm(formula=formula, data=df_results,  groups=df_results["repetition"]).fit()
mod1.summary()
# %%
