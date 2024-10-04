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
cwd = pathlib.Path(__file__).parent
def display_shap(cwd, model_name):
    shap_values, X, y = pickle.load(
    io.open(
        cwd.parent /
        model_name,
        'rb'))
    shap.plots.bar(shap_values, max_display=160)
    plt.show()
# %%
display_shap(cwd, 'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_05.pkl')
# %%
display_shap(cwd, 'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_10.pkl')
# %%
display_shap(cwd, 'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_15.pkl')
# %%
display_shap(cwd, 'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_20.pkl')
# %%
display_shap(cwd, 'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_25.pkl')
# %%
display_shap(cwd, 'model-data/output/T20240808-p_model_fi_shap_ft_set_vif_all.pkl')
# %%
