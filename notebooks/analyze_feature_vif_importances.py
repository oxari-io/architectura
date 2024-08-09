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
        'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_15.pkl',
        'rb'))

shap_values


# %%
# ax = plt.gca()
shap.summary_plot(shap_values, X, show=True)
# %%
shap.plots.bar(shap_values, max_display=160)
# %%
cwd = pathlib.Path(__file__).parent
shap_values, X, y = pickle.load(
    io.open(
        cwd.parent /
        'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_05.pkl',
        'rb'))

shap_values


# %%
# ax = plt.gca()
# shap.summary_plot(shap_values, X, show=True, max_display=30)
# %%
shap.plots.bar(shap_values, max_display=160)

# %%

# %%
cwd = pathlib.Path(__file__).parent
shap_values, X, y = pickle.load(
    io.open(
        cwd.parent /
        'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_10.pkl',
        'rb'))

shap_values


# %%
# ax = plt.gca()
# shap.summary_plot(shap_values, X, show=True, max_display=30)
# %%
shap.plots.bar(shap_values, max_display=160)
# %%
