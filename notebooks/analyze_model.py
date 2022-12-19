# %%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import seaborn as sns
from IPython.display import display
import pathlib
import pickle
import tracemalloc
from memory_profiler import profile
import pickletools
import inspect
# https://towardsdatascience.com/the-power-of-pickletools-handling-large-model-pickle-files-7f9037b9086b
# %%
cwd = pathlib.Path(__file__).parent
meta_model = pickle.load((cwd.parent/'local/objects/MetaModel_19-12-2022_prod.pkl').open('rb'))
# %%
all_op_codes = list(pickletools.genops((cwd.parent/'local/objects/MetaModel_19-12-2022_prod.pkl').open('rb')))
all_op_codes
# %%
inspect.getsource(meta_model)
# %%
