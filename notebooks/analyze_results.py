# %%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import seaborn as sns
from IPython.display import display
import pathlib
# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/results_sequential_02.csv', index_col=0)
df_results["scope_estimator"] = pd.Categorical(df_results["scope_estimator"])
df_results
# %%
list(df_results.columns.values)
# %%
sns.boxplot(data=df_results[df_results!="DummyEstimator"], x="sMAPE", y="scope_estimator")
# %%
sns.histplot(data=df_results[df_results!="DummyEstimator"], x="sMAPE", hue="scope_estimator", kde=True)

# %%
formula = "sMAPE ~ imputer * preprocessor * feature_selector"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["scope_estimator"]).fit()
mod1.summary()
# %%
formula = "sMAPE ~ imputer + preprocessor + feature_selector"
mod1 = smf.mixedlm(formula=formula, data=df_results, groups=df_results["scope_estimator"]).fit()
mod1.summary()
# %%
formula = "sMAPE ~ imputer + preprocessor + feature_selector + scope_estimator"
mod1 = smf.glm(formula=formula, data=df_results, family=sm.families.Gaussian()).fit()
mod1.summary()


# %%