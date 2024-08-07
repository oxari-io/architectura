# %%
import pathlib
from alibi.explainers import plot_ale, plot_pd, plot_pd_variance, plot_permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.patheffects as path_effects
import cloudpickle as pickle
import shap
import io
import numpy as np
# %%
DATA = "T20240422"

# %%
cwd = pathlib.Path(__file__).parent
shap_values, X, y = pickle.load(
    io.open(
        cwd.parent /
        'model-data/output/T20240505_p_model_experiment_feature_impact_explainer_shap.pkl',
        'rb'))

shap_values


# %%
ax = plt.gca()
shap.summary_plot(shap_values, X, show=True, max_display=30)
# %%
ranked = pd.DataFrame(shap_values.abs.values.sum(0),
                      columns=["shap-value"],
                      index=shap_values.feature_names).sort_values(
                          "shap-value", ascending=True)
ax = plt.plot(ranked.values)
plt.title("Importance")
plt.xlabel("Feature Index")
plt.ylabel("Sum of shap values")
# %%
print("Least important features according to shap")
ranked[:20]

# %%
shap.plots.scatter(shap_values[:, 'ft_numc_revenue'])

# %%
shap.plots.scatter(shap_values[:, 'ft_numc_revenue'],
                   color=shap_values[:, 'ft_numc_total_assets'])
# %%
# # Here only 1000 observations are visualized, because this plot is quite heavy
# # and can crash your Jupyter Notebook
shap.plots.force(shap_values[:50])
# %%
shap.plots.waterfall(shap_values[0])
# %%
shap.plots.waterfall(shap_values[1])
# %%
shap.plots.waterfall(shap_values[2])
# %%
shap.plots.bar(shap_values, max_display=160)

# %%
shap.plots.heatmap(shap_values)

# %%
pd_importance, X, y = pickle.load(
    io.open(
        cwd.parent /
        'model-data/output/T20240501_p_model_experiment_feature_impact_explainer_pd.pkl',
        'rb'))
pd_importance
# %%

feature_names = [ft for ft in pd_importance.meta['params']['feature_names']]
feature_indices = [i for i, ft in enumerate(pd_importance.meta['params']['feature_names']) if not ft.startswith('ft_cat')]
num_features = len(feature_indices)
num_cols = 4
is_even = (num_features % num_cols) == 0
ncols = num_cols
nrows = (num_features //
         num_cols) if is_even else (num_features // num_cols) + 1

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

for i, ft in enumerate(feature_indices):
    ax = axes.flatten()[i]
    # plot_ale(ale_values,sharey=None, ax=ax)
    feature_values = pd_importance.data['feature_values'][ft]
    pd_values = pd_importance.data['pd_values'][ft][0]
    ax.plot(feature_values, pd_values)
    ax.set_title(f'{feature_names[ft]}')
    # ax.set_xscale('symlog')
    # ax.set_yscale('symlog')

plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 100))
plot_pd(pd_importance, features=feature_indices, ax=ax, sharey=None)
plt.show()
# %%
# %%
cwd = pathlib.Path(__file__).parent
permut_values, X, y = pickle.load(
    io.open(
        cwd.parent /
        'model-data/output/T20240502_p_model_experiment_feature_impact_explainer_permut.pkl',
        'rb'))

# Some plots explained https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454
# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 20))
plot_permutation_importance(permut_values, ax=ax)
plt.show()

# %%
cwd = pathlib.Path(__file__).parent
ale_values, X, y = pickle.load(
    io.open(
        cwd.parent /
        'model-data/output/T20240429_p_model_experiment_feature_impact_explainer_ale.pkl',
        'rb'))

# Some plots explained https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454
# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
plot_ale(ale_values, features=list(range(0,6)), ax=ax, sharey=None)
plt.show()
# %%
feature_names = ale_values.data['feature_names']
num_features = len(feature_names)
num_cols = 3
is_even = (num_features % num_cols) == 0
ncols = num_cols
nrows = (num_features //
         num_cols) if is_even else (num_features // num_cols) + 1

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
constant = False

for i, ft in enumerate(feature_names):
    ax = axes.flatten()[i]
    x = ale_values.data['feature_values'][i]
    y = ale_values.data['ale_values'][i].flatten() #+ constant * ale_values.constant_value
    ax.plot(x,y)
    ax.scatter(x,y, s=10)
    ax.set_title(f'{ft}')
    ax.set_xscale('log')
    ax.set_yscale('symlog')

plt.show()

# %%
# %%
# pdv_importance, X, y = pickle.load(
#     io.open(
#         cwd.parent /
#         'model-data/output/T20240116_p_model_experiment_feature_impact_explainer_pdv.pkl',
#         'rb'))
# pdv_importance
# # %%
# fig, ax = plt.subplots(1, 1, figsize=(10, 20))
# plot_pd_variance(pdv_importance, ax=ax)

# TODO: Compare different XGB model
# TODO: Compare different FastSVR with SVR model

# %%
