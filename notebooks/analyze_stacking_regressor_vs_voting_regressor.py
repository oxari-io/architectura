# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_stacking_regressor_vs_voting_regressor.csv', index_col=0)
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

# %%
plt.figure(figsize=(15,5))
fig = sns.boxplot(results, x="scope_estimator", y="time", hue="scope")
add_median_labels(fig)
plt.title('time vs scope_estimator')
plt.xlabel('Scope Estimator')
plt.xticks(rotation=30)
plt.ylabel('time')
plt.legend(title = 'scope')

# %%
# Fit the ANOVA model
results['raw_sMAPE'] = results['raw.sMAPE']
model = ols('raw_sMAPE ~ C(scope) * C(scope_estimator)', data=results[results['scope_estimator'] != 'BaselineEstimator']).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Display ANOVA table
print("ANOVA Table:")
print(anova_table)

# %%
# Perform Tukey's HSD post-hoc test
results['scope'] = results['scope'].astype(str)
results['scope_estimator'] = results['scope_estimator'].astype(str)

# Create a new column for interaction
results['interaction_group'] = results['scope'] + '_' + results['scope_estimator']

# Perform Tukey's HSD post-hoc test for interaction
interaction_tukey = pairwise_tukeyhsd(results['raw.sMAPE'], results['interaction_group'])
print("\nTukey's HSD Post-Hoc Test for Interaction:")
print(interaction_tukey)

# %%
results['raw_sMAPE'] = results['raw.sMAPE']
data = results[results['scope_estimator'] != 'BaselineEstimator']
data['interaction_group'] = data['scope'] + '_' + data['scope_estimator']
md = smf.mixedlm("raw_sMAPE ~ C(scope_estimator)", data, groups=data['interaction_group'])
mdf = md.fit()
print(mdf.summary())
# %%
