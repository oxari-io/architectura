# %%
import sys

sys.path.append("..")

# import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.preprocessing import KBinsDiscretizer
from base import ( OxariDataManager)
# from base.common import OxariLoggerMixin
# from base.confidence_intervall_estimator import BaselineConfidenceEstimator
# from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager,PreviousScopeFeaturesDataManager
from datasources.online import S3Datasource,CachingS3Datasource
from datasources.local import LocalDatasource
from base.run_utils import get_default_datamanager_configuration
from base.dataset_loader import (CategoricalLoader, CompanyDataFilter, Datasource, FinancialLoader, OxariDataManager, PartialLoader, ScopeLoader)
from datasources.loaders import RegionLoader
# from feature_reducers import AgglomerateFeatureReducer, PCAFeatureReducer, FactorAnalysisFeatureReducer, GaussRandProjectionFeatureReducer, IsomapDimensionalityFeatureReducer, SparseRandProjectionFeatureReducer, ModifiedLocallyLinearEmbeddingFeatureReducer
# from imputers import RevenueQuantileBucketImputer
# from pipeline.core import DefaultPipeline
# from preprocessors import IIDPreprocessor
# from scope_estimators import SupportVectorEstimator
import missingno as msno
from datasources.loaders import NetZeroIndexLoader

# %%
# dataset = DefaultDataManager(scope_loader=S3ScopeLoader(), financial_loader=S3FinancialLoader(), categorical_loader=S3CategoricalLoader()).run()
# dataset = DefaultDataManager(S3Datasource(path='model-input-data/scopes_auto.csv'),
#                              S3Datasource(path='model-input-data/financials_auto.csv'),
#                              S3Datasource(path='model-input-data/categoricals_auto.csv'),
#                              other_loaders=[NetZeroIndexLoader()]).run()
# dataset = PreviousScopeFeaturesDataManager().run()
dataset = PreviousScopeFeaturesDataManager(
        FinancialLoader(datasource=LocalDatasource(path="../model-data/input/financials_auto.csv")),
        ScopeLoader(datasource=LocalDatasource(path="../model-data/input/scopes_auto.csv")),
        CategoricalLoader(datasource=LocalDatasource(path="../model-data/input/categoricals_auto.csv")),
        RegionLoader(),).run()

DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
SPLIT_1 = bag.scope_1
SPLIT_2 = bag.scope_2
SPLIT_3 = bag.scope_3

# %%
DATA

# %%
X, y = SPLIT_1.train
X
# %%
X_numeric = X.filter(regex="ft_num", axis=1)
X_numeric
# %% [markdown]
# ## Missing Values within the Features
# ### Matrix of missing values
# On a sample of 500
# Dark means values exist
msno.matrix(X.sample(500))

# %% [markdown]
# ### Bar char of missing value counts
# On a sample of 500
msno.bar(X.sample(500), sort="descending")

# %% [markdown]
# ### Matrix of missing values
# On all data correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another
msno.heatmap(X)
# %% [markdown]
# ### Matrix of missing values
# The dendrogram allows you to more fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap.
# To interpret this graph, read it from a top-down perspective. Cluster leaves which linked together at a distance of zero fully predict one another's presence—one variable might always be empty when another is filled, or they might always both be filled or both empty, and so on.
msno.dendrogram(X)
# %% [markdown]
# ## Features of the dataset
sns.pairplot(X_numeric.sample(frac=0.05), dropna=True, kind='scatter', corner=True)

# %%
df_melted = X_numeric.melt(var_name='column')
g = sns.FacetGrid(df_melted, col='column', col_wrap=2, sharex=False, sharey=False, height=6)
g.map(sns.kdeplot, 'value')
# %%
df_melted = X_numeric.melt(var_name='column')
g = sns.FacetGrid(df_melted, col='column', col_wrap=2, sharex=False, sharey=False, height=6)
g.map(sns.histplot, 'value')
# %%
