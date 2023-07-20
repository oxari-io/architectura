# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent/'local/eval_results/experiment_static_feature_selection.csv', index_col=0)
# %%
plt.figure(figsize=(17,10))
fig = sns.boxplot(
    data = results, 
    x = "list_name",
    y = "raw.sMAPE",
    hue = "scope"
)
plt.title('raw.sMAPE vs list_name')
plt.xlabel('list_name')
plt.ylabel('raw.sMAPE')
plt.legend(title = 'scope')

# %%
plt.figure(figsize=(17,10))
fig = sns.boxplot(
    data = results, 
    x = "list_name",
    y = "time",
    hue = "scope"
)
plt.title('time vs list_name')
plt.xlabel('list_name')
plt.ylabel('time')
plt.legend(title = 'scope')
