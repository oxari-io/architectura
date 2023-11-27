# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects

#%%
def add_median_labels(ax, fmt='.2f'):
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
# Filter rows where scope_estimator is MiniModelArmyEstimator
filtered_df = results[results['scope_estimator'] == 'MiniModelArmyEstimator']

# Extract the relevant columns for plotting
weights_columns = [col for col in results.columns if col.startswith('weights.')]

# Melt the DataFrame to make it suitable for seaborn's boxplot
melted_df = pd.melt(filtered_df, id_vars=['scope_estimator'], value_vars=weights_columns,
                    var_name='weight_type', value_name='weight_value')

# Extract the bucket type from the weight_type column
melted_df['bucket_type'] = melted_df['weight_type'].apply(lambda x: x.split('.')[-1])
melted_df['bucket_number'] = melted_df['weight_type'].apply(lambda x: x.split('.')[-2].split('_')[-1])

# Set up the plotting area
plt.figure(figsize=(12, 8))

# Use seaborn's boxplot to visualize the weights based on the type of bucket
fig = sns.boxplot(x='bucket_type', y='weight_value', hue='bucket_number', data=melted_df)
add_median_labels(fig)

# Customize the plot
plt.title('Weights based on Bucket Type for MiniModelArmyEstimator')
plt.xlabel('Bucket Type')
plt.ylabel('Weight Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Weight Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()


# %%
# Filter rows where scope_estimator is AlternativeCVMiniModelArmyEstimator
filtered_df = results[results['scope_estimator'] == 'AlternativeCVMiniModelArmyEstimator']

# Extract the relevant columns for plotting
weights_columns = [col for col in results.columns if col.startswith('weights.')]

# Melt the DataFrame to make it suitable for seaborn's boxplot
melted_df = pd.melt(filtered_df, id_vars=['scope_estimator'], value_vars=weights_columns,
                    var_name='weight_type', value_name='weight_value')

# Extract the bucket type from the weight_type column
melted_df['bucket_type'] = melted_df['weight_type'].apply(lambda x: x.split('.')[-1])
melted_df['bucket_number'] = melted_df['weight_type'].apply(lambda x: x.split('.')[-2].split('_')[-1])

# Set up the plotting area
plt.figure(figsize=(12, 8))

# Use seaborn's boxplot to visualize the weights based on the type of bucket
fig = sns.boxplot(x='bucket_type', y='weight_value', hue='bucket_number', data=melted_df)
add_median_labels(fig)

# Customize the plot
plt.title('Weights based on Bucket Type for AlternativeCVMiniModelArmyEstimator')
plt.xlabel('Bucket Type')
plt.ylabel('Weight Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Weight Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()
# %%
