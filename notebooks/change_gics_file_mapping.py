# %%
from IPython.core.pylabtools import figsize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %%
dataset = pd.read_csv('../preprocessors/helper/gics.csv')
dataset
# %%
dataset = dataset.ffill()
dataset
# %%
modified_dataset = dataset.rename(columns={
    "Unnamed: 1":"ft_catm_sector_name",
    "Unnamed: 3":"ft_catm_industry_group_name",
    "Unnamed: 5":"ft_catm_industry_name",
    "Unnamed: 7":"ft_catm_sub_industry_name",
}).iloc[:, :-2]
modified_dataset
# %%
modified_dataset.to_csv('../preprocessors/helper/gics_mod.csv')
# %%
