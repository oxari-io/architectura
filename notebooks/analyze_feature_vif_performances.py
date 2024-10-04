# %%
import sys


sys.path.append("..")
import pathlib
from IPython.display import display
from pmdarima.metrics import smape

from datasources.loaders import RegionLoader
from datasources.local import LocalDatasource
from base.dataset_loader import CategoricalLoader, CompanyDataFilter, FinancialLoader, OxariDataManager, ScopeLoader
import matplotlib.pyplot as plt
from base.common import OxariMetaModel
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

DATE = "T20240808"
model_vif_05 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_05.pkl", 'rb'))
model_vif_10 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_10.pkl", 'rb'))
model_vif_15 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_15.pkl", 'rb'))
model_vif_20 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_20.pkl", 'rb'))
model_vif_25 = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_25.pkl", 'rb'))
model_vif_all = pkl.load(io.open(PARENT_PATH+f"/model-data/output/{DATE}-p_model_fi_ft_set_vif_all.pkl", 'rb'))
model_vif_20_no_stocks = pkl.load(io.open(PARENT_PATH+f"/model-data/output/T20240816-p-model-python-3.10.13.pkl", 'rb'))
# %%
bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
DATA_SCOPE_1 = bag.scope_1
X, y = DATA_SCOPE_1.test
X_no_prior = X.drop(columns=["ft_numc_prior_tg_numc_scope_1","ft_numc_prior_tg_numc_scope_2", "ft_numc_prior_tg_numc_scope_3"])
# %%
res = []
all_models = [model_vif_05,model_vif_10,model_vif_15,model_vif_20,model_vif_25, model_vif_all, model_vif_20_no_stocks]
all_model_names = ["model_vif_05","model_vif_10","model_vif_15","model_vif_20","model_vif_25","model_vif_all", "model_vif_20_no_stocks"]
for m, m_name in zip(all_models,all_model_names):
    for d, d_name in zip([X, X_no_prior],["has_prior_year", "without_prior_year"]):
        res.append((m_name, d_name, smape(m.predict(d)["scope_1"], y)))

res
# %%
df_res = pd.DataFrame(res, columns=["model", "dataset", "smape"])
df_res_pivot = df_res.pivot(index="model", columns="dataset")
# sns.lineplot(df_res, x="model", y="smape", hue="dataset")
# %%
print(df_res_pivot)
#                        smape                   
# dataset       has_prior_year without_prior_year
# model                                          
# model_vif_05       12.202460          21.836750
# model_vif_10       10.025841          16.843655
# model_vif_15        9.420325          15.386200
# model_vif_20        8.574679          14.046274
# model_vif_25        8.475433          14.081915
# model_vif_all      22.581010          33.842102

# %%
model_vif_25.collect_eval_results()
# %%
results = []
for model, model_name in zip(all_models, all_model_names):
    results.append(pd.json_normalize([{"model_name":model_name, **r} for r in model.collect_eval_results()]))
pd.concat(results).T.to_csv(PARENT_PATH + f'/local/prod_runs/VIF_model_pipelines_{DATE}.csv')

# %%
