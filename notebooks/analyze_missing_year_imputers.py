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
df_results["mae"] = df_results["overall.MAE"]
df_results["smape"] = df_results["overall.sMAPE"]
df_results
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="scope_estimator", y="raw.sMAPE")
sns.boxplot(df_results, x="imputer", y="smape", hue="difficulty_level")
# %%
plt.figure(figsize=(15,10))
sns.boxplot(df_results, x="difficulty_level", y="smape", hue="imputer")
plt.show()
# %%
plt.figure(figsize=(10,5))
sns.kdeplot(df_results[(df_results.scope==1) & (df_results.type=="delinked")], x="raw.sMAPE", hue="scope_estimator")
plt.show()
# %%
formula = "smape ~ f1 + scope_estimator + scope"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["groups"]).fit()
mod1.summary()
# %%plt.figure(figsize=(10,10))
formula = "smape ~ f1 + scope_estimator + scope + type"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["groups"]).fit()
mod1.summary()