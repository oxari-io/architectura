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
# %%
cwd = pathlib.Path(__file__).parent
df_results = pd.read_csv(cwd.parent/'local/eval_results/experiment_confidence_estimator_performance.csv', index_col=0)
# df_results["scope_estimator"] = pd.Categorical(df_results["scope_estimator"])
df_results
# %%
df = df_results.dropna()
sns.scatterplot(df, y="mean_coverage_ground_truth", x="median_range", hue="ci_estimator")
# %%
