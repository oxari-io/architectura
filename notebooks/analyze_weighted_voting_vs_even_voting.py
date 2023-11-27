# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects

#%%
def add_median_labels(ax, fmt='.1f'):
    """Credits: https://stackoverflow.com/a/63295846/4865723
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

# %%
# See full results of the experiment 
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_weighted_voting_vs_even_voting.csv', index_col=0)
results

# %%
plt.figure(figsize=(15,5))
fig = sns.boxplot(results, x="scope_estimator", y="raw.sMAPE", hue="scope")
add_median_labels(fig)
plt.title('raw.sMAPE vs scope_estimator')
plt.xlabel('Scope Estimator')
plt.xticks(rotation=30)
plt.ylabel('raw.sMAPE')
plt.legend(title = 'scope')

#%%
# Filter DataFrame based on scope_estimator
mma_results = results[results['scope_estimator'] == 'MiniModelArmyEstimator']

# Select relevant columns
weights_columns = mma_results.filter(like='weights.bucket_')

# Melt the DataFrame to reshape it for easier plotting
melted_df = pd.melt(mma_results, id_vars=['scope_estimator'], value_vars=weights_columns.columns, var_name='Weight_Type', value_name='Weight_Value')

# Plot using seaborn
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
fig = sns.boxplot(x='scope_estimator', y='Weight_Value', hue='Weight_Type', data=melted_df)
add_median_labels(fig)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Weights based on scope_estimator')
plt.xlabel('scope_estimator')
plt.ylabel('Weight Value')
plt.show()

#%%
# Filter DataFrame based on scope_estimator
cv_mma_results = results[results['scope_estimator'] == 'AlternativeCVMiniModelArmyEstimator']

# Select relevant columns
weights_columns = cv_mma_results.filter(like='weights.bucket_')

# Melt the DataFrame to reshape it for easier plotting
melted_df = pd.melt(cv_mma_results, id_vars=['scope_estimator'], value_vars=weights_columns.columns, var_name='Weight_Type', value_name='Weight_Value')

# Plot using seaborn
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
fig = sns.boxplot(x='scope_estimator', y='Weight_Value', hue='Weight_Type', data=melted_df)
add_median_labels(fig)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Weights based on scope_estimator')
plt.xlabel('scope_estimator')
plt.ylabel('Weight Value')
plt.show()
# %%
