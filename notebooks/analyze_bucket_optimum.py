# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_bucket_optimum.csv', index_col=0)
df_results["mae"] = df_results["raw.MAE"]
df_results["smape"] = df_results["raw.sMAPE"]

# %%
# sns.lineplot(data=df_results, x="test.n_buckets", y="smape")
sns.lineplot(data=df_results, x="test.n_buckets", y="time")
# %%