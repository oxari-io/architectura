# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_AAA.csv', index_col=0).dropna(subset=["raw.sMAPE"])
df_results["scope_estimator"] = pd.Categorical(df_results["scope_estimator"])
df_results = df_results[~df_results["feature_selector"].str.startswith("PCA")]
df_results
# %%
sns.boxplot(data=df_results[df_results!="DummyEstimator"], x="raw.sMAPE", y="scope_estimator")
# %%
sns.histplot(data=df_results[df_results!="DummyEstimator"], x="raw.sMAPE", hue="scope_estimator", kde=True)
# %%
res = df_results.sort_values("raw.sMAPE")[['imputer', 'fin_transformer',
       'cat_transformer', 'scope_transformer', 'preprocessor',
       'feature_selector', 'scope_estimator']+df_results.filter(regex="^(raw|train|test)").columns.tolist()].head(50)#.iloc[0].head(50)
# %%
res.iloc[0][["test.sMAPE", "train.sMAPE", "raw.sMAPE"]]
# %%
