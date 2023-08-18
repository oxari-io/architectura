# %%
import pandas as pd
from pathlib import Path

ROOT = Path("model-data/input/")
KEYS = ["key_isin", "key_year"]

scopes_old = pd.read_csv(ROOT/"scopes_auto.csv")
scopes_new = pd.read_csv(ROOT/"scopes_auto_new.csv")
financials_old = pd.read_csv(ROOT/"financials_auto_old.csv")
financials_new = pd.read_csv(ROOT/"financials_auto.csv")
# %%
financials_old.merge(scopes_old, left_on=KEYS, right_on=KEYS)
# %%
financials_new.merge(scopes_old, left_on=KEYS, right_on=KEYS)

# %%
financials_new.merge(scopes_new, left_on=KEYS, right_on=KEYS)
# %%
financials_old.merge(scopes_new, left_on=KEYS, right_on=KEYS)

# %%
len(financials_new.key_isin.unique())
# %%
len(financials_old.key_isin.unique())

# %%
