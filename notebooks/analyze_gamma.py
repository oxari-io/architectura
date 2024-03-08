# %%
import sys

sys.path.append("..")

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from base import (LocalDataSaver, LocalLARModelSaver, LocalMetaModelSaver, OxariDataManager, OxariMetaModel, OxariSavingManager, helper)
from base.common import OxariLoggerMixin
from base.confidence_intervall_estimator import BaselineConfidenceEstimator
from base.helper import LogTargetScaler
from datasources.core import DefaultDataManager
from datasources.online import S3Datasource
from feature_reducers import AgglomerateFeatureReducer, PCAFeatureReducer, FactorAnalysisFeatureReducer, GaussRandProjectionFeatureReducer, IsomapDimensionalityFeatureReducer, SparseRandProjectionFeatureReducer, ModifiedLocallyLinearEmbeddingFeatureReducer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import SupportVectorEstimator
import missingno as msno
from datasources.loaders import NetZeroIndexLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import GammaRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import TransformedTargetRegressor

# %%
# dataset = DefaultDataManager(scope_loader=S3ScopeLoader(), financial_loader=S3FinancialLoader(), categorical_loader=S3CategoricalLoader()).run()
dataset = DefaultDataManager(S3Datasource(path='model-input-data/scopes.csv'),
                             S3Datasource(path='model-input-data/financials.csv'),
                             S3Datasource(path='model-input-data/categoricals.csv'),
                             other_loaders=[NetZeroIndexLoader()]).run()
# dataset = PreviousScopeFeaturesDataManager().run()
DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
SPLIT_1 = bag.scope_1
SPLIT_2 = bag.scope_2
SPLIT_3 = bag.scope_3
# %%
X, y = SPLIT_1.train
X
# %%
X_numeric = X.filter(regex="ft_num", axis=1)
X_numeric
# %%
column = "ft_numd_employees"
X_numeric_target = X_numeric.dropna(how="any",subset=[column])
X_, y_ = X_numeric_target[X_numeric_target.columns.difference([column])], X_numeric_target[column]
# %% [markdown]
# ## Missing Values within the Features
model = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), PolynomialFeatures(3), TransformedTargetRegressor(GammaRegressor(alpha=0.1, max_iter = 2000), transformer=MinMaxScaler((0.00001, 1))))
X_train, X_test, y_train, y_test = train_test_split(X_, y_)
model.fit(X_train, y_train) 

# %%
results = pd.DataFrame([y_test.values, model.predict(X_test)], index="true pred".split()).T
sns.kdeplot(results.iloc[:, 1])
# %%
results = cross_val_score(model, X_, y_, cv=3, scoring="neg_mean_absolute_error")
results
# %%
