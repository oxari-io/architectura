# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
X_var = "y_true"
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_optuna_metric.csv', index_col=0)
results["standardized_residuals"] = (results["residual"] - results["residual"].mean())/results["residual"].std()
results["buckets"] = pd.qcut(results[X_var], 5)
# IMPORTANT: The residuals are small because they are standardized
# %%
results['abs_residuals'] = results['residual'].abs()
results.groupby('metric')['abs_residuals'].agg(['median', 'mean', 'std', 'sum', 'min', 'max'])

# %%
plt.figure(figsize=(10, 5))
fig = sns.scatterplot(
    data=results,
    x=X_var,
    y="abs_residuals",
    hue='metric'
)
fig.set_xlabel('Y-True')
fig.set_ylabel('Residuals')
fig.set_title('predicted value vs residuals')
plt.xscale('log')
plt.xticks(rotation=90)
# plt.yscale('log')
plt.show()
# %%
plt.figure(figsize=(10, 5))
fig = sns.boxplot(
    data=results,
    # x="buckets",
    y="abs_residuals",
    x='metric'
)
fig.set_xlabel('Pred Value')
fig.set_ylabel('Residuals')
fig.set_title('predicted value vs residuals')
# plt.xscale('log')
plt.yscale('log')
plt.xticks(rotation=90)
plt.show()
# %%
plt.figure(figsize=(10, 5))
plt.xticks(rotation=90)
fig = sns.boxplot(
    data=results,
    x="buckets",
    y="abs_residuals",
    hue='metric'
)
# plt.xscale('log')
plt.yscale('log')
fig.set_xlabel('Pred Value')
fig.set_ylabel('Residuals')
fig.set_title('predicted value vs residuals')
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.xticks(rotation=90)
fig = sns.scatterplot(
    data=results,
    x=X_var,
    y="abs_residuals",
    hue='metric'
)
plt.xscale('log')
plt.yscale('log')
fig.set_xlabel('Pred Value')
fig.set_ylabel('Residuals')
fig.set_title('predicted value vs residuals')
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.xticks(rotation=90)
fig = sns.scatterplot(
    data=results,
    x=X_var,
    y="abs_residuals",
    hue='metric'
)
# plt.xscale('log')
# plt.yscale('log')
fig.set_xlabel('Pred Value')
fig.set_ylabel('Residuals')
fig.set_title('predicted value vs residuals')
plt.show()

# %%
