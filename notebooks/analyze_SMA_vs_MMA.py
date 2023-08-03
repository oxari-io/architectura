# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
# See full results of the experiment for SCOPE 1
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_SMA_vs_MMA.csv', index_col=0)
results["balanced_acc"] = results["test.classifier.balanced_accuracy"].fillna(0)
results["smape"] = results["raw.sMAPE"]
results.head()

# %%
# Raw sMAPE (DSMA vs SMA vs MMA)

plt.figure(figsize=(10, 6))
fig = sns.boxplot(
    data=results, 
    x='scope', 
    y='raw.sMAPE',
    hue='scope_estimator'
)
add_median_labels(fig)
plt.title('raw.sMAPE value for each estimator')
plt.legend(title='Estimator')
plt.xlabel('scope')
plt.ylabel('raw.sMAPE value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# %%
# See raw sMAPE and balanced accuracy - focus on estimators

plt.figure(figsize=(10, 6))
fig = sns.scatterplot(
    data=results, 
    x='test.classifier.balanced_accuracy', 
    y='raw.sMAPE',
    hue='scope_estimator'
)
plt.title('raw.sMAPE vs balanced accuracy')
plt.legend(title='Estimator')
plt.xlabel('balanced accuracy')
plt.ylabel('raw.sMAPE value')
plt.xticks(rotation=45, ha='right')

plt.figure(figsize=(10, 6))
fig = sns.kdeplot(
    data=results, 
    x='test.classifier.balanced_accuracy', 
    y='raw.sMAPE',
    hue='scope_estimator'
)
plt.title('raw.sMAPE vs balanced accuracy')

plt.xlabel('balanced accuracy')
plt.ylabel('raw.sMAPE value')
plt.xticks(rotation=45, ha='right')


# See raw sMAPE and balanced accuracy - focus on scope

plt.figure(figsize=(10, 6))
fig = sns.scatterplot(
    data=results, 
    x='test.classifier.balanced_accuracy', 
    y='raw.sMAPE',
    hue='scope'
)
plt.title('raw.sMAPE vs balanced accuracy')
plt.legend(title='Scope')
plt.xlabel('balanced accuracy')
plt.ylabel('raw.sMAPE value')
plt.xticks(rotation=45, ha='right')

plt.figure(figsize=(10, 6))
fig = sns.kdeplot(
    data=results, 
    x='test.classifier.balanced_accuracy', 
    y='raw.sMAPE',
    hue='scope'
)
plt.title('raw.sMAPE vs balanced accuracy')

plt.xlabel('balanced accuracy')
plt.ylabel('raw.sMAPE value')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()

# %%
# %%
formula = "smape ~ balanced_acc + repetition"
mod1 = smf.mixedlm(formula=formula, data=results[results['scope_estimator'] == "MiniModelArmyEstimator"], groups="scope").fit()
# mod1 = smf.glm(formula=formula, data=results[results['scope_estimator'] == "MiniModelArmyEstimator"]).fit()
mod1.summary()
# %%
# ax = plt.gca()
# ax.scatter(mod1.mu, mod1.resid_pearson)
# ax.hlines(0, 0, 1)
# ax.set_xlim(0, 1)
# ax.set_title('Residual Dependence Plot')
# ax.set_ylabel('Pearson Residuals')
# ax.set_xlabel('Fitted values')
# %%
