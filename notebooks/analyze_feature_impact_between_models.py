# %%
import sys

sys.path.append("..")
import pathlib
from IPython.display import display
from pmdarima.metrics import smape

from datasources.loaders import RegionLoader
from datasources.local import LocalDatasource
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, FinancialLoader, ScopeLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from base import OxariDataManager
from datasources.core import DefaultDataManager, PreviousScopeFeaturesDataManager
from datasources.online import S3Datasource
from pathlib import Path
import missingno as msno
import matplotlib.patches as mpatches
sns.set_palette('viridis')

PARENT_PATH = Path('..').absolute().resolve().as_posix()
PARENT_PATH
# %%
dataset = PreviousScopeFeaturesDataManager(
    FinancialLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/financials.csv")),
    ScopeLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/scopes.csv")),
    CategoricalLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/categoricals.csv")),
    RegionLoader(),
).set_filter(CompanyDataFilter(frac=1)).run()
DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
DATA
# %%
import cloudpickle as pkl 
import io 
model_full = pkl.load(io.open(PARENT_PATH+"/model-data/output/T20240805_p_model-si_python-3.10.13.pkl", 'rb'))
model_small = pkl.load(io.open(PARENT_PATH+"/model-data/output/T20240805_p_model_python-3.10.13.pkl", 'rb'))
# %%
bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
DATA_SCOPE_1 = bag.scope_1
X, y = DATA_SCOPE_1.test
X_small = X[model_small.pipelines['scope_1'].feature_selector.selected_features_]
X_no_prior = X.drop(columns=["ft_numc_prior_tg_numc_scope_1","ft_numc_prior_tg_numc_scope_2", "ft_numc_prior_tg_numc_scope_3"])
X_no_prior_small = X_small.drop(columns=["ft_numc_prior_tg_numc_scope_1","ft_numc_prior_tg_numc_scope_2", "ft_numc_prior_tg_numc_scope_3"])
# %%
pred_small = model_small.predict(X_small)
pred_full = model_full.predict(X_small)
pred_small_all_data = model_small.predict(X)
pred_full_all_data = model_full.predict(X)
pred_small_no_prior = model_small.predict(X_no_prior)
pred_full_no_prior = model_full.predict(X_no_prior)
pred_small_no_prior_subset = model_small.predict(X_no_prior_small)
pred_full_no_prior_subset = model_full.predict(X_no_prior_small)
# %%
# %%
# %%
# %%
# %%
# %%

# %%
# %%
smape(pred_small_no_prior_subset["scope_1"], y)
# %%
smape(pred_full_no_prior_subset["scope_1"], y)

# %%
pd.DataFrame([
    ("online-model", "full-data", "has_prior_year", smape(pred_small_all_data["scope_1"], y)),
    ("imputation-model", "full-data", "has_prior_year", smape(pred_full_all_data["scope_1"], y)),
    ("online-model", "subset-data", "has_prior_year", smape(pred_small["scope_1"], y)),
    ("imputation-model", "subset-data", "has_prior_year", smape(pred_full["scope_1"], y)),
    ("online-model", "full-data", "without_prior_year", smape(pred_small_no_prior["scope_1"], y)),
    ("imputation-model", "full-data", "without_prior_year", smape(pred_full_no_prior["scope_1"], y)),
    ("online-model", "subset-data", "without_prior_year", smape(pred_small_no_prior_subset["scope_1"], y)),
    ("imputation-model", "subset-data", "without_prior_year", smape(pred_full_no_prior_subset["scope_1"], y)),
], columns=["model", "dataset", "uses_prior_scope", "smape"])
# %%
