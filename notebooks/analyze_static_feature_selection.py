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
add_median_labels(fig)
plt.title('raw.sMAPE vs list_name')
plt.xlabel('list_name')
plt.ylabel('raw.sMAPE')
plt.legend(title = 'scope')

# %%
plt.figure(figsize=(17,10))
fig = sns.boxplot(
    data = results[results['time'] < 5000], 
    x = "list_name",
    y = "time",
    hue = "scope"
)
add_median_labels(fig)
plt.title('time vs list_name')
plt.xlabel('list_name')
plt.ylabel('time')
plt.legend(title = 'scope')

# %%
plt.figure(figsize=(17,10))
fig = sns.scatterplot(
    data=results[results['time'] < 5000],
    x='time', 
    y='raw.sMAPE', 
    hue='list_name', 
    style='scope', 
    s=100
)
plt.title('time vs raw.sMAPE - all scopes')
plt.xlabel('time')
plt.ylabel('raw.sMAPE')

# %%
plt.figure(figsize=(17,10))
fig = sns.scatterplot(
    data=results[(results['time'] < 5000) & (results['scope'] == 1)],
    x='time', 
    y='raw.sMAPE', 
    hue='list_name', 
    # style='scope', 
    s=100
)
plt.title('time vs raw.sMAPE - scope 1')
plt.xlabel('time')
plt.ylabel('raw.sMAPE')

# %%
plt.figure(figsize=(17,10))
fig = sns.scatterplot(
    data=results[(results['time'] < 5000) & (results['scope'] == 2)],
    x='time', 
    y='raw.sMAPE', 
    hue='list_name', 
    # style='scope', 
    s=100
)
plt.title('time vs raw.sMAPE - scope 2')
plt.xlabel('time')
plt.ylabel('raw.sMAPE')

# %%
plt.figure(figsize=(17,10))
fig = sns.scatterplot(
    data=results[(results['time'] < 5000) & (results['scope'] == 3)],
    x='time', 
    y='raw.sMAPE', 
    hue='list_name', 
    style='scope', 
    s=100
)
plt.title('time vs raw.sMAPE - scope 3')
plt.xlabel('time')
plt.ylabel('raw.sMAPE')