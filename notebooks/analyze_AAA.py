# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_AAA.csv', index_col=0)
df_results["scope_estimator"] = pd.Categorical(df_results["scope_estimator"])
df_results
# %%
list(df_results.columns.values)
# %%
sns.boxplot(data=df_results[df_results!="DummyEstimator"], x="raw.sMAPE", y="scope_estimator")
# %%
sns.histplot(data=df_results[df_results!="DummyEstimator"], x="raw.sMAPE", hue="scope_estimator", kde=True)
