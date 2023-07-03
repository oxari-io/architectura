# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_optuna_metric_10_reps.csv', index_col=0)
results["standardized_residuals"] = (results["residual"] - results["residual"].mean())/results["residual"].std()
results
# %%
results_smape = results[results['metric'] == 'smape']
results_smape
# %%
# %%
plt.figure(figsize=(10, 5))
plt.xticks(rotation=90)
fig = sns.scatterplot(
    data=results[results["y_pred"]],
    x="y_pred",
    y="standardized_residuals",
    hue='metric'
)
plt.xscale('log')
# plt.yscale('log')
fig.set_xlabel('Pred Value')
fig.set_ylabel('Residuals')
fig.set_title('smape vs residuals')
# %%

results_msle = results[results['metric'] == 'mean_squared_log_error']
# %%
msle_val_X = results_msle['val_X'].tolist()
residuals_msle = results_msle['residuals'].tolist()
# %%
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
fig = sns.scatterplot(x=msle_val_X, y=residuals_msle)
fig.set_xlabel('predicted_values')
fig.set_ylabel('residuals')
fig.set_title('msle vs residuals')
# %%

results_mse = results[results['metric'] == 'mean_squared_error']
# %%
mse_val_X = results_mse['val_X'].tolist()
residuals_mse = results_mse['residuals'].tolist()
# %%
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
fig = sns.scatterplot(x=mse_val_X, y=residuals_mse)
fig.set_xlabel('predicted_values')
fig.set_ylabel('residuals')
fig.set_title('mse vs residuals')