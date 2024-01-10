# %%
import sys

sys.path.append("..")
import pathlib
from IPython.display import display

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
import autoimpute as imp
import missingno as msno
import matplotlib.patches as mpatches
sns.set_palette('viridis')

PARENT_PATH = Path('..').absolute().resolve().as_posix()
PARENT_PATH
# %%
dataset = PreviousScopeFeaturesDataManager(
    FinancialLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/financials_auto.csv")),
    ScopeLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/scopes_auto.csv")),
    CategoricalLoader(datasource=LocalDatasource(path=PARENT_PATH + "/model-data/input/categoricals_auto.csv")),
    RegionLoader(),
).set_filter(CompanyDataFilter(frac=1)).run()
DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
DATA
# %%
def visualize_matrix(df, title="Missingness Matrix"):
    fig = plt.figure()
    ax = msno.matrix(df)
    ax.set_title(title)
    ax.set_ylabel('Data point')
    ax.set_xlabel('Features')
    fig.tight_layout()
    plt.show()

visualize_matrix(DATA.filter(regex="^ft_num", axis=1), 'Missingness: Numeric Freatures')
visualize_matrix(DATA.filter(regex="^ft_cat", axis=1), 'Missingness: Categorical Features')
visualize_matrix(DATA.filter(regex="^tg_", axis=1), 'Missingness: Targets')

# %%
df = DATA.filter(regex="^ft_num", axis=1)
msno.bar(df)
plt.show()
# %%
# NOTE: If we have ppe we might not have the others based on this image
msno.heatmap(df)
plt.show()

# %%
msno.dendrogram(df)
plt.show()
# %%
