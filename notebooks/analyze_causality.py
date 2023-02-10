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
# %%
X, y = SPLIT_1.train
X
# %%
numerical_data: pd.DataFrame = X.iloc[:, X.columns.str.startswith("ft_num")]
numerical_data
# %%
without_missing = numerical_data.dropna(how="any")
with_missing = numerical_data[numerical_data.isnull().any(1)]

# %%
import cdt
from cdt.independence.graph import FSGNN as IndependenceTester
import networkx as nx

glasso = IndependenceTester()
skeleton = glasso.predict(without_missing)
nx.draw(skeleton, with_labels=True)
# %%
new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
nx.draw(new_skeleton)

# %%
model = cdt.causality.graph.GES()
output_graph = model.predict(without_missing, new_skeleton)
nx.draw(output_graph)
# %%
output_graph.nodes.data()
# %%
# pos = nx.spectral_layout(output_graph)
nx.draw_planar(output_graph, with_labels=True)
# %%
# TRY OUT MULTIPLE ALGORITHMS
# https://towardsdatascience.com/causal-discovery-6858f9af6dcb
glasso = cdt.independence.graph.Glasso()
df = without_missing
skeleton = glasso.predict(df)
nx.draw(skeleton)
# %%
# # PC algorithm
# model_pc = cdt.causality.graph.PC()
# graph_pc = model_pc.predict(df, skeleton)
# fig=plt.figure(figsize=(15,10))
# pos = nx.spring_layout(graph_pc, k=0.15, iterations=20)
# nx.draw(graph_pc, pos, with_labels=True)
# %%
# GES algorithm
model_ges = cdt.causality.graph.GES()
graph_ges = model_ges.predict(df, skeleton)
fig = plt.figure(figsize=(15, 10))
pos = nx.spring_layout(graph_ges, k=0.4, iterations=10)
nx.draw(graph_ges, pos, with_labels=True)

# %%
# LiNGAM Algorithm
model_lingam = cdt.causality.graph.LiNGAM()
graph_lingam = model_lingam.predict(df)
fig = plt.figure(figsize=(15, 10))
pos = nx.spring_layout(graph_lingam, k=0.4, iterations=10)
nx.draw(graph_lingam, pos, with_labels=True)
# %%
print(graph_lingam.edges)
# %%
# https://stackoverflow.com/a/69368605
from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import KNNImputer, SimpleImputer

imputer = SimpleImputer().fit(numerical_data)
numerical_data_filled = pd.DataFrame(imputer.transform(numerical_data), columns=numerical_data.columns)
discretizer = KBinsDiscretizer(10, encode='ordinal').fit(numerical_data_filled)
discrete_df = pd.DataFrame(discretizer.transform(df), columns=numerical_data.columns)
discrete_df
# %%
bn = BayesianNetwork(graph_lingam.edges)
bn.fit(discrete_df, estimator=BayesianEstimator, prior_type="BDeu")
bn.get_cpds()
# %%
print(bn.get_cpds('ft_numc_market_cap'))

# %%
from pgmpy.inference import VariableElimination

infer = VariableElimination(bn)
g_dist = infer.query(['ft_numc_revenue'])
print(g_dist)
# %%
chosen_companies = with_missing[(with_missing.isnull().sum(axis=1) < 2).values]
chosen_companies
# %%
non_graph_cols = ["ft_numc_stock_return", "ft_numc_equity", "ft_numd_employees", "ft_numc_roa", "ft_numc_roe"]
nan_mask = chosen_companies.isnull()
discrete_chosen = pd.DataFrame(discretizer.transform(imputer.transform(chosen_companies)), columns=numerical_data.columns)
discrete_chosen[nan_mask.values] = None
in_graph_discrete_chosen = discrete_chosen[discrete_chosen.columns.difference(non_graph_cols)]
in_graph_discrete_chosen
# %%
dict_chosen = in_graph_discrete_chosen.to_dict(orient='records')
dict_chosen
# %%
g_dist = infer.query(['ft_numc_rd', 'ft_numc_cash'], evidence={ 'ft_numc_inventories': 0.0, 'ft_numc_market_cap': 5.0, 'ft_numc_net_income': 5.0,'ft_numc_total_liab': 1.0})
print(g_dist)
# %%
from pgmpy.models import LinearGaussianBayesianNetwork
bn = LinearGaussianBayesianNetwork(graph_lingam.edges)
bn.fit(df)
bn.get_cpds()
# %%
