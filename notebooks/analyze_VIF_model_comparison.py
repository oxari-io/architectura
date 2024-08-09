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

DATE = "T20240807"
model_vif_05 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_05.pkl", 'rb'))
model_vif_10 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_10.pkl", 'rb'))
model_vif_15 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_15.pkl", 'rb'))
model_vif_15 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_15.pkl", 'rb'))
model_vif_15 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_15.pkl", 'rb'))
# %%
bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
DATA_SCOPE_1 = bag.scope_1
X, y = DATA_SCOPE_1.test
X_no_prior = X.drop(columns=["ft_numc_prior_tg_numc_scope_1","ft_numc_prior_tg_numc_scope_2", "ft_numc_prior_tg_numc_scope_3"])
# %%
res = []
for m, m_name in zip([model_vif_05,model_vif_10,model_vif_15],["model_vif_05","model_vif_10","model_vif_15"]):
    for d, d_name in zip([X, X_no_prior],["has_prior_year", "without_prior_year"]):
        res.append((m_name, d_name, smape(m.predict(d)["scope_1"], y)))

res
# %%
pd.DataFrame(res, columns=["model", "dataset", "smape"])
# %%

#           model             dataset      smape
# 0  model_vif_05      has_prior_year  12.553558
# 2  model_vif_10      has_prior_year  10.450130
# 4  model_vif_15      has_prior_year   9.540122
# 1  model_vif_05  without_prior_year  22.356073
# 3  model_vif_10  without_prior_year  17.047266
# 5  model_vif_15  without_prior_year  15.237889
