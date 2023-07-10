# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_optuna_metric.csv', index_col=0)
results["standardized_residuals"] = (results["residual"] - results["residual"].mean())/results["residual"].std()
# IMPORTANT: The residuals are small because they are standardized
# %%
results['abs_residuals'] = results['standardized_residuals'].abs()
results.groupby('metric').sum()['abs_residuals']

# %%
plt.figure(figsize=(10, 5))
plt.xticks(rotation=90)
fig = sns.scatterplot(
    data=results[(results["abs_residuals"] > 40)],
    x="y_pred",
    y="abs_residuals",
    hue='metric'
)
plt.xscale('log')
# plt.yscale('log')
fig.set_xlabel('Pred Value')
fig.set_ylabel('Residuals')
fig.set_title('predicted value vs residuals')

# %%
plt.figure(figsize=(10, 5))
plt.xticks(rotation=90)
fig = sns.scatterplot(
    data=results,
    x="y_pred",
    y="abs_residuals",
    hue='metric'
)
plt.xscale('log')
plt.yscale('log')
fig.set_xlabel('Pred Value')
fig.set_ylabel('Residuals')
fig.set_title('predicted value vs residuals')

# %%
plt.figure(figsize=(10, 5))
plt.xticks(rotation=90)
fig = sns.scatterplot(
    data=results,
    x="y_pred",
    y="abs_residuals",
    hue='metric'
)
# plt.xscale('log')
# plt.yscale('log')
fig.set_xlabel('Pred Value')
fig.set_ylabel('Residuals')
fig.set_title('predicted value vs residuals')

# %%
