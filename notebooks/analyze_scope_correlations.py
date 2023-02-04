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
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from postprocessors.scope_imputers import ScopeImputerPostprocessor
from datasources.core import PreviousScopeFeaturesDataManager
from base.dataset_loader import OxariDataManager
from datasources import LocalDatasource
from IPython.display import display, Markdown, Latex
pd.set_option('display.float_format', lambda x: '%.5f' % x)
# pd.reset_option('^display.', silent=True)



# %%
cwd = pathlib.Path(__file__).parent
meta_model = pickle.load((cwd.parent / 'local/objects/meta_model/snapshot-03-02-2023.pkl').open('rb'))
meta_model
# %%
dataset = PreviousScopeFeaturesDataManager(
    LocalDatasource(path=cwd.parent/'local/data/scopes_auto.csv'),
    LocalDatasource(path=cwd.parent/'local/data/financials_auto.csv'),
    LocalDatasource(path=cwd.parent/'local/data/categoricals_auto.csv'),
).run()
DATA = dataset.get_data_by_name(OxariDataManager.ORIGINAL)
DATA
# %%
print("Impute scopes with Model")
scope_imputer = ScopeImputerPostprocessor(estimator=meta_model).run(X=DATA).evaluate()
scope_imputer
# %%
data_merged_with_input = scope_imputer.data.merge(DATA, on=["key_year", "key_isin"])
data_merged_with_input

# %% [markdown]
# ## Dataset
# ### Description of the training Dataset 
# This table describes key characteristics of the dataset used for training. 
display(Markdown(f"The data contained {DATA.shape[0]} data points with {DATA.shape[1]} columns."))
display(Markdown(f"Of the seventeen columns {DATA.columns.str.startswith('tg_').sum()} acted as dependent variables, {DATA.columns.str.startswith('ft_').sum()} as independent variables and {DATA.columns.str.startswith('key_').sum()} as columns with meta information."))
display(DATA.describe().T)
# %% [markdown]
# ### Description of the Dataset imputed
display(Markdown("ATTENTION: This correlation analysis is based on a subset of data."))
display(Markdown("The data was selected based on two criteria."))
display(Markdown("1. Data without information about any of the 3 scope targets where removed."))
display(Markdown("2. Data from years prior 2016 where removed."))
display(data_merged_with_input.describe().T)
# %% [markdown]
# ### Amount of imputed data counts
# The amount of data points that where imputed.
display(scope_imputer.data[["predicted_s1", "predicted_s2", "predicted_s3"]].sum())
# %% [markdown]
# ## Correlation Analysis
# ### No predictions
display("Scope 1 Correlation (no predictions)")
display(data_merged_with_input[data_merged_with_input["predicted_s1"]==False][["tg_numc_scope_1_x", "ft_numc_revenue"]].corr())
display("Scope 2 Correlation (no predictions)")
display(data_merged_with_input[data_merged_with_input["predicted_s2"]==False][["tg_numc_scope_2_x", "ft_numc_revenue"]].corr())
display("Scope 3 Correlation (no predictions)")
display(data_merged_with_input[data_merged_with_input["predicted_s3"]==False][["tg_numc_scope_3_x", "ft_numc_revenue"]].corr())

# %% [markdown]
# ### Only predictions
display("Scope 1 Correlation (only predictions)")
display(data_merged_with_input[data_merged_with_input["predicted_s1"]==True][["tg_numc_scope_1_x", "ft_numc_revenue"]].corr())
display("Scope 2 Correlation (only predictions)")
display(data_merged_with_input[data_merged_with_input["predicted_s2"]==True][["tg_numc_scope_2_x", "ft_numc_revenue"]].corr())
display("Scope 3 Correlation (only predictions)")
display(data_merged_with_input[data_merged_with_input["predicted_s3"]==True][["tg_numc_scope_3_x", "ft_numc_revenue"]].corr())

# %% [markdown]
# ### Overall
display("Scope 1 Correlation (overall)")
display(data_merged_with_input[["tg_numc_scope_1_x", "ft_numc_revenue"]].corr())
display("Scope 2 Correlation (overall)")
display(data_merged_with_input[["tg_numc_scope_2_x", "ft_numc_revenue"]].corr())
display("Scope 3 Correlation (overall)")
display(data_merged_with_input[["tg_numc_scope_3_x", "ft_numc_revenue"]].corr())