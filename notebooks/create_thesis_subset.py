# %%
from IPython.core.pylabtools import figsize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %%
dataset = pd.read_csv('dataset_combined_2.csv')
dataset 
# %%
isin_counts = dataset.dropna(subset=['tg_numc_scope_1','tg_numc_scope_2','tg_numc_scope_3'],how='all').key_ticker.value_counts()
isin_counts
# %%
print(f"Amount of companies: {len(isin_counts)}")
# %%
minimum_years = dataset.dropna(subset=['tg_numc_scope_1','tg_numc_scope_2','tg_numc_scope_3'],how='all').groupby("key_ticker")["key_year"].min()
minimum_years
# %%
maximum_years = dataset.dropna(subset=['tg_numc_scope_1','tg_numc_scope_2','tg_numc_scope_3'],how='all').groupby("key_ticker")["key_year"].max()
maximum_years
# %%
def count_missing_years(df):
    """
    Count the missing years for each company in the provided dataframe.
    
    Args:
    - df (pd.DataFrame): DataFrame with columns 'key_ticker' and 'key_year'.
    
    Returns:
    - pd.Series: Number of missing years for each company.
    """
    # Determine the minimum and maximum years for each company
    year_range = df.groupby('key_ticker')['key_year'].agg([min, max])

    # Find missing years for each company
    def find_missing_years(group):
        min_year, max_year = year_range.loc[group.name]
        full_range = set(range(min_year, max_year + 1))
        actual_years = set(group['key_year'].values)
        return len(full_range - actual_years)

    return df.groupby('key_ticker').apply(find_missing_years)

missing_years = count_missing_years(dataset.dropna(subset=['tg_numc_scope_1','tg_numc_scope_2','tg_numc_scope_3'],how='all'))
missing_years
# %%
df = missing_years.to_frame().merge(minimum_years.to_frame(), on="key_ticker").merge(maximum_years.to_frame(), on="key_ticker").rename(columns={0: 'missing_years', "key_year_x":"min", "key_year_y":"max"})
df["range"] = df["min"].astype(str) + "-" + df["max"].astype(str)
df
# %%
fig, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(15, 20))
sns.histplot(isin_counts, discrete=True, ax=ax1)
sns.histplot(df, x="missing_years", ax=ax2, discrete=True)
sns.histplot(df, x="range", ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, ha='center')
ax1.set_title('Number of years available per company')
ax2.set_title('Number of companies with a "hole" in the set of years')
ax3.set_title('Number of companies with a certain range of available years')
ax1.set_xlabel('Years available')
fig.tight_layout()
# fig.title()
plt.show()
# %%
print("We provide 5 companies with 3 years, 1 with 8, 1 with 14")
isin_counts[(isin_counts == 2)]
# %%
isin_counts[(isin_counts == 3)]
# %%
isin_counts[(isin_counts == 7)]
# %%
isin_counts[(isin_counts == 8)]
# %%
isin_counts[(isin_counts == 14)]
# %%
two_set = ["US92214X1063", "JP3274400005", "ZAE000216537"]
three_set = ["GB00BY9D0Y18", "US00773T1016", "DE0006231004", "TW0002823002"]
mid_set = ["ES0113307062", "GB00BJVJZD68"]
all_set = ["US0584981064"]
# %%
all_combined = two_set + three_set + mid_set + all_set
all_combined
# %%
dataset[dataset.key_ticker.isin(all_combined)].iloc[:,2:].to_csv('hemanth_subset.csv')
# %%
