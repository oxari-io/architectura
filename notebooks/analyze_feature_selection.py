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
from base.helper import LogarithmScaler
from datasources.core import DefaultDataManager
from datasources.online import S3Datasource
from feature_reducers import AgglomerateFeatureReducer, PCAFeatureReducer, FactorAnalysisFeatureReducer, GaussRandProjectionFeatureReducer, IsomapDimensionalityFeatureReducer, SparseRandProjectionFeatureReducer, ModifiedLocallyLinearEmbeddingFeatureReducer
from imputers import RevenueQuantileBucketImputer
from pipeline.core import DefaultPipeline
from preprocessors import IIDPreprocessor
from scope_estimators import SupportVectorEstimator
# %%
# dataset = DefaultDataManager(scope_loader=S3ScopeLoader(), financial_loader=S3FinancialLoader(), categorical_loader=S3CategoricalLoader()).run()
dataset = DefaultDataManager(S3Datasource(path='model-input-data/scopes_auto.csv'), S3Datasource(path='model-input-data/financials_auto.csv'),
                             S3Datasource(path='model-input-data/categoricals_auto.csv')).run()
# dataset = PreviousScopeFeaturesDataManager().run()
DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
SPLIT_1 = bag.scope_1
SPLIT_2 = bag.scope_2
SPLIT_3 = bag.scope_3

dp1 = DefaultPipeline(
    preprocessor=IIDPreprocessor(),
    feature_reducer=IsomapDimensionalityFeatureReducer(10),
    imputer=RevenueQuantileBucketImputer(buckets_number=5),
    scope_estimator=SupportVectorEstimator(),
    ci_estimator=BaselineConfidenceEstimator(),
    scope_transformer=LogarithmScaler(),
).optimise(*SPLIT_1.train).fit(*SPLIT_1.train).evaluate(*SPLIT_1.rem, *SPLIT_1.val).fit_confidence(*SPLIT_1.train)

# %%
import itertools as it


def visualize(X, y, in_3d=False, **kwargs):
    X_reduced = dp1.feature_selector.transform(X)
    n_dim = dp1.feature_selector.n_components_
    if not in_3d:
        ll = set(it.combinations(range(n_dim), 2))
        cols = 3
        rows = (len(ll) // cols) + 1 if (len(ll) % cols) != 0 else (len(ll) // cols)
        figsize = kwargs.pop('figsize', (cols*5, rows*5))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        y = y
        for dims, ax in zip(ll, axes.flatten()):
            for g in np.unique(y):
                ix = (y == g).values.flatten()
                xc, yc = X_reduced[ix].iloc[:, list(dims)].values.T
                ax.scatter(xc, yc, label=g, alpha=0.25, s=75)
                ax.set_xlabel(f"Dim {dims[0]}")
                ax.set_ylabel(f"Dim {dims[1]}")
                # xc, yc, zc = X_reduced[ix, dims].T
                # ax.scatter3D(xc, yc, zc, label = g, s = 100)
    else:
        ll = set(it.combinations(range(n_dim), 3))
        cols = 3
        rows = (len(ll) // cols) + 1 if (len(ll) % cols) != 0 else (len(ll) // cols)
        figsize = kwargs.pop('figsize', (cols*5, rows*5))
        fig, axes = plt.subplots(rows, cols, figsize=figsize, subplot_kw={'projection':'3d'})
        y = y
        for dims, ax in zip(ll, axes.flatten()):
            for g in np.unique(y):
                ix = (y == g).values.flatten()
                xc, yc, zc = X_reduced[ix].iloc[:, list(dims)].values.T
                ax.scatter3D(xc, yc, zc, label = g, s = 100)
                ax.set_xlabel(f"Dim {dims[0]}")
                ax.set_ylabel(f"Dim {dims[1]}")
                ax.set_zlabel(f"Dim {dims[2]}")
    ax.legend()
    fig.tight_layout()
    plt.show()


discretizer = KBinsDiscretizer(encode='ordinal', strategy='kmeans')
X, y = SPLIT_1.test
X = dp1.preprocessor.transform(X, y)

filter_criterion = y > 0
in_3d = True
visualize(X[filter_criterion], pd.qcut(y, 10)[filter_criterion], in_3d=in_3d)

# %%

# TODO: Visualise function in this notebook should be method of FeatureReducer class
# TODO: Put filter_criterion and in_3d configurations into the method parameter
# TODO: Experiment which saves a plot for each feature reducers visualisation 2D and 3D
