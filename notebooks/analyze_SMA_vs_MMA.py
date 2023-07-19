# %%
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# %%
# See full results of the experiment for SCOPE 1
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_SMA_vs_MMA.csv', index_col=0)
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

plt.tight_layout()
