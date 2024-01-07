# %%
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.dataset_loader import CategoricalLoader, FinancialLoader, ScopeLoader

from datasources.core import PreviousScopeFeaturesDataManager
from datasources.loaders import RegionLoader
from datasources.local import LocalDatasource 

import pandas as pd
from base import OxariDataManager
from base.run_utils import get_small_datamanager_configuration
from sklearn.model_selection import train_test_split
import imputers.equilibrium_method as equi

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# %%
all_results = []
difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dataset = PreviousScopeFeaturesDataManager(
        FinancialLoader(datasource=LocalDatasource(path="../model-data/input/financials_auto.csv")),
        ScopeLoader(datasource=LocalDatasource(path="../model-data/input/scopes_auto.csv")),
        CategoricalLoader(datasource=LocalDatasource(path="../model-data/input/categoricals_auto.csv")),
        RegionLoader(),
    ).run()

# %%
bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
SPLIT_1 = bag.scope_1
X, Y = SPLIT_1.train

# %%
X_new = X.copy()
X_train, X_test = train_test_split(X_new, test_size=0.5)
keeping_criterion_2 = (X_test.isna().mean(axis=0)<0.5)
keep_columns_2 = X_train.loc[:, keeping_criterion_2].columns

# %%
import importlib
importlib.reload(equi)

imputer_2: equi.EquilibriumImputer = equi.EquilibriumImputer(verbose=False, max_iter=100, diff_tresh=0.01, mims_tresh=0.001, max_diff_increase_thresh=0.6).clone()
X_subset = X_train[keep_columns_2]
imputer_2 = imputer_2.fit(X_subset)

# %%
imputer_2.evaluate(X_test[keep_columns_2], p=0.1)
imputer_2.visualize()
plt.show()











# %%
imputer_2.transform(X_test[keep_columns_2])
# %%
imputer_2.visualize()

# %%
