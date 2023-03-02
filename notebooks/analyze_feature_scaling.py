# %%
import pathlib
from tkinter import Y

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_feature_scaling.csv', index_col=0)
df_results
# %%
df_results["configuration"] = pd.Categorical(df_results["fin_transformer"]+"-"+df_results["scope_transformer"])
df_results["groups"] = pd.Categorical(df_results["scope_transformer"]+df_results["repetition"].astype(str))
df_results["scope"] = pd.Categorical(df_results["scope"])
# df_results["scope-type"] = pd.Categorical(df_results["scope"].astype(str)+"-"+df_results["type"])
df_results["repetition"] = pd.Categorical(df_results["repetition"])
df_results["smape"] = df_results["raw.sMAPE"]
df_results["f1"] = df_results["test.classifier.balanced_f1"]
df_results
# %%
plt.figure(figsize=(10,5))
sns.boxplot(df_results, x="smape", y="fin_transformer", hue="scope_transformer")

plt.show()
# %%
formula = "smape ~ fin_transformer + scope_transformer"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["repetition"]).fit()
mod1.summary()
# %%
sns.scatterplot(df_results, x="f1", y="smape")