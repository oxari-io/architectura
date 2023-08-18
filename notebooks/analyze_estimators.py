# %%
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_estimators.csv', index_col=0)


# %%
grouped_data = results.groupby(['scope_estimator', 'scope'])['test.sMAPE'].mean().reset_index()

plt.figure(figsize=(10, 6))
fig = sns.barplot(
    data=grouped_data, 
    x='scope_estimator', 
    y='test.sMAPE', 
    hue='scope'
)
plt.title('test.sMAPE value for each estimator')
plt.xlabel('Estimator')
plt.ylabel('test.sMAPE Value')
plt.legend(title='Scope')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# %%
grouped_data = results.groupby(['scope_estimator', 'scope'])['raw.sMAPE'].mean().reset_index()

plt.figure(figsize=(10, 6))
fig = sns.barplot(
    data=grouped_data, 
    x='scope_estimator', 
    y='raw.sMAPE', 
    hue='scope'
)
plt.title('raw.sMAPE value for each estimator')
plt.xlabel('Estimator')
plt.ylabel('raw.sMAPE Value')
plt.legend(title='Scope')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# %%
closer_results = results[results['scope_estimator'].isin(['AdaboostEstimator', 'MiniModelArmyEstimator'])]

grouped_data = closer_results.groupby(['scope_estimator', 'scope'])['test.sMAPE'].mean().reset_index()

plt.figure(figsize=(10, 6))
fig = sns.barplot(
    data=grouped_data, 
    x='scope_estimator', 
    y='test.sMAPE', 
    hue='scope'
)
plt.title('test.sMAPE value for each estimator')
plt.xlabel('Estimator')
plt.ylabel('test.sMAPE Value')
plt.legend(title='Scope')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# %%
closer_results = results[results['scope_estimator'].isin(['AdaboostEstimator', 'MiniModelArmyEstimator'])]

plt.figure(figsize=(10, 6))
fig = sns.boxplot(
    data=closer_results, 
    x='scope_estimator', 
    y='raw.sMAPE', 
    hue='scope'
)

add_median_labels(fig)

plt.title('raw.sMAPE value for each estimator')
plt.xlabel('estimator')
plt.ylabel('raw.sMAPE Value')
plt.legend(title='Scope')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# %%
