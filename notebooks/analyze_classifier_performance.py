# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_classifier_performance.csv', index_col=0)
df_results["groups"] = pd.Categorical(df_results["scope_estimator"]+"-Scope"+df_results["scope"].astype(str))
df_results
# %%
sns.boxplot(df_results, x="scope_estimator", y="raw.sMAPE")
# sns.boxplot(df_results, x="scope_estimator", y="raw.sMAPE", hue="scope")
# %%
plt.figure(figsize=(10,10))
sns.scatterplot(df_results, x="test.classifier.balanced_f1", y="raw.sMAPE", hue="groups")
# %%
