# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results_delinked = pd.read_csv(cwd.parent/'local/eval_results/experiment_classifier_performance_delinked.csv', index_col=0)
df_results_linked = pd.read_csv(cwd.parent/'local/eval_results/experiment_classifier_performance_linked.csv', index_col=0)
df_results_linked["type"] = "linked"
df_results_delinked["type"] = "delinked"
df_results = pd.concat([df_results_linked, df_results_delinked])
df_results
# %%
df_results["groups"] = pd.Categorical(df_results["scope_estimator"]+"-Scope"+df_results["scope"].astype(str)+"-"+df_results["type"])
df_results["scope"] = pd.Categorical(df_results["scope"])
df_results["scope-type"] = pd.Categorical(df_results["scope"].astype(str)+"-"+df_results["type"])
df_results["repetition"] = pd.Categorical(df_results["repetition"])
df_results["smape"] = df_results["raw.sMAPE"]
df_results["f1"] = df_results["test.classifier.balanced_f1"]
df_results
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="scope_estimator", y="raw.sMAPE")
sns.boxplot(df_results, x="scope_estimator", y="raw.sMAPE", hue="scope")
# %%
plt.figure(figsize=(15,10))
# sns.scatterplot(df_results, x="test.classifier.balanced_f1", y="raw.sMAPE", hue="scope-type")
sns.scatterplot(df_results[df_results["type"]=="delinked"], x="test.classifier.balanced_f1", y="raw.sMAPE", hue="groups")
# sns.scatterplot(df_results[df_results["type"]=="linked"], x="test.classifier.balanced_f1", y="raw.sMAPE", hue="scope")
plt.show()
# %%
plt.figure(figsize=(10,5))
# sns.kdeplot(df_results[(df_results.scope==1)], x="raw.sMAPE", hue="scope_estimator")
sns.kdeplot(df_results[(df_results.scope==1) & (df_results.type=="delinked")], x="raw.sMAPE", hue="scope_estimator")
# sns.histplot(df_results[(df_results.scope==1) & (df_results.type=="delinked")], x="raw.sMAPE", hue="scope_estimator", bins=25)
# sns.kdeplot(df_results[(df_results.scope==1) & (df_results.type=="linked")], x="raw.sMAPE", hue="scope_estimator")
plt.show()
# %%
formula = "smape ~ f1 + scope_estimator + scope"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["groups"]).fit()
mod1.summary()
# %%plt.figure(figsize=(10,10))
formula = "smape ~ f1 + scope_estimator + scope + type"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["groups"]).fit()
mod1.summary()