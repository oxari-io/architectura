# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/test.csv', index_col=0)
df_results["scope_estimator"] = pd.Categorical(df_results["scope_estimator"])
df_results
# %%
list(df_results.columns.values)
# %%
sns.boxplot(data=df_results[df_results!="DummyEstimator"], x="test.sMAPE", y="feature_selector")
# %%
sns.boxplot(data=df_results[df_results!="DummyEstimator"], x="test.R2", y="feature_selector")
# %%
sns.boxplot(data=df_results[df_results!="DummyEstimator"], x="test.MAE", y="feature_selector")
# %%
sns.boxplot(data=df_results[df_results!="DummyEstimator"], x="test.RMSE", y="feature_selector")
# %%
sns.boxplot(data=df_results[df_results!="DummyEstimator"], x="test.MAPE", y="feature_selector")
# %%
