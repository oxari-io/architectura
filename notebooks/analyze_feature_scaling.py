# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.concat([
    # pd.read_csv(cwd.parent/'local/eval_results/experiment_feature_scaling_1.csv', index_col=0),
    pd.read_csv(cwd.parent/'local/eval_results/experiment_feature_scaling.csv', index_col=0),
])
df_results = df_results[df_results["feature_selector"] == "DummyFeatureReducer"]
# %%
df_results["configuration"] = pd.Categorical(df_results["fin_transformer"]+"-"+df_results["scope_transformer"])
df_results["groups"] = pd.Categorical(df_results["scope_transformer"]+df_results["scope"].astype(str))
df_results["session"] = pd.Categorical(df_results["repetition"].astype(str)+"-"+df_results["scope"].astype(str))
df_results["scope"] = pd.Categorical(df_results["scope"])
# df_results["scope-type"] = pd.Categorical(df_results["scope"].astype(str)+"-"+df_results["type"])
df_results["repetition"] = pd.Categorical(df_results["repetition"])
df_results["smape"] = df_results["raw.sMAPE"]
df_results["f1"] = df_results["test.classifier.balanced_f1"]
df_results
# %%
plt.figure(figsize=(10,10))
sns.boxplot(df_results, x="smape", y="configuration", hue="scope")

plt.show()
# %%
plt.figure(figsize=(10,10))
sns.pointplot(df_results, x="scope", y="smape", hue="configuration")
plt.show()
# %%
plt.figure(figsize=(10,5))
ax = sns.boxplot(df_results, x="configuration", y="smape")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha="center")
plt.show()
# %%
df_results.groupby(["fin_transformer", "scope_transformer"])["smape"].describe().drop(columns="count").style.highlight_min(color = 'blue',  
                       axis = 0).highlight_max(color = 'darkred',  
                       axis = 0)

# %%
plt.figure(figsize=(10,5))
ax = sns.boxplot(df_results, x="fin_transformer", y="smape")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, ha="center")
plt.show()
# %%
plt.figure(figsize=(10,5))
ax = sns.scatterplot(df_results, x="scope_transformer", y="smape")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, ha="center")
plt.show()
# %%
pivoted_scopes = df_results.pivot(columns="scope", values="smape", index=["repetition", "feature_selector", "fin_transformer", "scope_transformer"])
pivoted_scopes.corr()
# %%
pivoted_feature_selectors = df_results.pivot(columns="feature_selector", values="smape", index=["repetition", "scope", "fin_transformer", "scope_transformer"])
pivoted_feature_selectors.head(50)
# %%
fig = plt.figure(figsize=(20, 5))
sns.lineplot(df_results, x=range(len(df_results)), y="smape", hue="scope")
plt.show()
# %%
# formula = "smape ~ feature_selector"
# mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["session"]).fit()
# # mod1 = smf.glm(formula=formula, data=df_results).fit()
# mod1.summary()
# # %%
# ax = plt.gca()
# ax.scatter(mod1.mu, mod1.resid_pearson)
# ax.hlines(0, 0, 1)
# ax.set_xlim(0, 1)
# ax.set_title('Residual Dependence Plot')
# ax.set_ylabel('Pearson Residuals')
# ax.set_xlabel('Fitted values')
# %%
# fig = sm.graphics.plot_regress_exog(mod1)
# fig.tight_layout(pad=1.0)
# %%
