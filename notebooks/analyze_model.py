# %%
import inspect
import pathlib
import pickle
# from memory_profiler import profile
import pickletools

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# https://towardsdatascience.com/the-power-of-pickletools-handling-large-model-pickle-files-7f9037b9086b
# %%
cwd = pathlib.Path(__file__).parent
meta_model = pickle.load((cwd.parent/'local/objects/MetaModel_20-12-2022_lightweight.pkl').open('rb'))
# %%
all_op_codes = list(pickletools.genops((cwd.parent/'local/objects/MetaModel_19-12-2022_prod.pkl').open('rb')))
[op[0] for op in all_op_codes]
# %%
inspect.getsource(meta_model)
# %%
def mock_data():
    num_data = {
        "stock_return": -0.03294267654418,
        "total_assets": 0.0,
        "ppe": 1446082.0,
        "year": 2019.0,
        "roa": 0.0145850556922605,
        "roe": 0.34,
        "total_liab": 1421.287,
        "equity": 1124.699,
        "revenue": 503.999999604178,
        "market_cap": 635.348579719647,
        "inventories": 13991.0,
        "net_income": 34.9999999725123,
        "cash": 231.043,
        "employees": 1000,
        "rd_expenses": 500,
        "isin": "FR0000051070",
        "scope_1": None,
        "scope_2": None,
        "scope_3": None,
    }

    cat_data = {
        "industry_name": "Industrial Conglomerates",
        # "company_name": "Aboitiz Equity Ventures Inc",
        "country_name": "Philippines",
        "sector_name": "Industrials",
    }
    df = pd.Series({**num_data, **cat_data}).to_frame().T.sort_index(axis=1)
    return df

print("\n", "Predict ALL on Mock data")
print(meta_model.predict(mock_data()))
# %%
