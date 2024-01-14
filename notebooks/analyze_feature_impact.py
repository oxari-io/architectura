# %%
import pathlib
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
cwd = pathlib.Path(__file__).parent
shap_values, X, y = pickle.load(io.open(cwd.parent / 'model-data/output/T20240113_p_model_experiment_feature_impact_explainer.pkl', 'rb'))

shap_values
# Some plots explained https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454
# %%
shap.summary_plot(shap_values, X, show=True, max_display=30)


# %%
shap.plots.scatter(shap_values[:, 'ft_numc_revenue'])

# %%
shap.plots.scatter(shap_values[:, 'ft_numc_revenue'], color=shap_values[:, 'ft_numc_total_assets'])
# %%
# # Here only 1000 observations are visualized, because this plot is quite heavy
# # and can crash your Jupyter Notebook
# shap.plots.force(shap_values[:100])
# %%
shap.plots.waterfall(shap_values[0])
# %%
shap.plots.waterfall(shap_values[1])
# %%
shap.plots.waterfall(shap_values[2])
# %%
shap.plots.bar(shap_values, max_display=30)

# %%
shap.plots.heatmap(shap_values)
# %%
pvd_importance, pvd_interaction = pickle.load(io.open(cwd.parent / 'model-data/output/T20231211_p_model_experiment_feature_impact_explainer_pvd.pkl', 'rb'))
pvd_importance
# %%
