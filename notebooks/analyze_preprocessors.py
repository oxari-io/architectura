# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects


# %%
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_preprocessors.csv', index_col=0)
results
# %%
plt.figure(figsize=(10,10))
ax = sns.boxenplot(results, x="preprocessor", y="raw.sMAPE")
plt.xticks(rotation=60)
plt.show()
 # %%
results.groupby("preprocessor")["raw.sMAPE"].describe().drop(columns="count").style.highlight_min(color = 'blue',  
                       axis = 0).highlight_max(color = 'darkred',  
                       axis = 0)
# %%
