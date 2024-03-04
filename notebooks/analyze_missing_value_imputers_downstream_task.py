# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_downstream_task.csv', index_col=0)
cat_rows = df_results["imputer"].str.startswith("Categorical")
df_results.loc[cat_rows, "imputer"] = (df_results.loc[cat_rows, "imputer"].astype(str) + ":" + df_results.loc[cat_rows, "cfg.preprocessor.reference"].astype(str)).values
df_results["mae"] = df_results["raw.MAE"]
df_results["smape"] = df_results["raw.sMAPE"]

# %%
# df_results["imputer"] = pd.Categorical(df_results["imputer"])
df_results["mae"] = df_results["raw.MAE"]
df_results["smape"] = df_results["raw.sMAPE"]
df_results
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="mae", y="imputer")
sns.boxplot(df_results, x="smape", y="imputer")
plt.gca().set_title("Imputer evaluation with PCA of 40")
plt.show()

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_downstream_task_validation.csv', index_col=0)
cat_rows = df_results["imputer"].str.startswith("Categorical")
df_results.loc[cat_rows, "imputer"] = (df_results.loc[cat_rows, "imputer"].astype(str) + ":" + df_results.loc[cat_rows, "cfg.preprocessor.reference"].astype(str)).values
df_results["mae"] = df_results["raw.MAE"]
df_results["smape"] = df_results["raw.sMAPE"]
# df_results
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="mae", y="imputer")
sns.boxplot(df_results, x="smape", y="imputer")
plt.gca().set_title("Imputer evaluation without dimensionality reduction")
plt.show()
# %%
plt.figure(figsize=(15,5))
# sns.boxplot(df_results, x="mae", y="imputer")
sns.boxplot(df_results, x="time", y="imputer")
plt.gca().set_title("Imputer evaluation without dimensionality reduction")
plt.show()


# # %%
# cwd = pathlib.Path(__file__).parent
# df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_missing_value_imputers_downstream_task_validation_2.csv', index_col=0)
# cat_rows = df_results["imputer"].str.startswith("Categorical")
# df_results.loc[cat_rows, "imputer"] = (df_results.loc[cat_rows, "imputer"].astype(str) + ":" + df_results.loc[cat_rows, "cfg.preprocessor.reference"].astype(str)).values
# df_results["mae"] = df_results["raw.MAE"]
# df_results["smape"] = df_results["raw.sMAPE"]
# # df_results
# # %%
# plt.figure(figsize=(15,5))
# # sns.boxplot(df_results, x="mae", y="imputer")
# sns.boxplot(df_results, x="smape", y="imputer")
# plt.gca().set_title("Imputer evaluation without dimensionality reduction")
# plt.show()

#  # %%
