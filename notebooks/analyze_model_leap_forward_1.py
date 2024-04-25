# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects
import numpy as np

# %%
cwd = pathlib.Path(__file__).parent
results = pd.read_csv(cwd.parent / 'local/eval_results/experiment_model_leap_forward_1.csv', index_col=0)
results["c_bucket_cls"] = np.where(results["optimal_params.cls.learning_rate"].isna(), "clf_rf", "clf_lgbm")
results["configuration"] = results["c_imputer"] + "-" + results["c_model"] + "-" + results["c_fintransformer"] + "-" + results["c_preprocessor"] + "-" + results["c_bucket_cls"]
# Hacking in forgotten reported value
results
# %%
plt.figure(figsize=(10,5))
ax = sns.boxplot(results, x="c_imputer", y="raw.sMAPE")
plt.show()
# %%
plt.figure(figsize=(10,5))
ax = sns.boxplot(results, x="c_model", y="raw.sMAPE")
plt.show()
# %%
plt.figure(figsize=(10,5))
ax = sns.boxplot(results, x="c_fintransformer", y="raw.sMAPE")
plt.show()
# %%
plt.figure(figsize=(10,5))
ax = sns.boxplot(results, x="c_preprocessor", y="raw.sMAPE")
plt.show()
# %%
plt.figure(figsize=(10,5))
ax = sns.boxplot(results, x="c_bucket_cls", y="raw.sMAPE")
plt.show()
 # %%
plt.figure(figsize=(12,10))
ax = sns.boxplot(results.sort_values("raw.sMAPE"), x="configuration", y="raw.sMAPE")
plt.xticks(rotation=90)
plt.show()
# %%
results.groupby("configuration")["raw.sMAPE"].describe().sort_values("50%").drop(columns=("count")).style.highlight_min(color = 'blue',  
                       axis = 0).highlight_max(color = 'darkred',  
                       axis = 0)

# %%
r_old = results[
    (results["c_imputer"]=="imputer_revenue_bucket_10") &
    (results["c_model"]=="default_weighting")  &
    (results["c_fintransformer"]=="ft_scaling_power") &
    (results["c_preprocessor"]=="preprocessor_iid") &
    (results["c_bucket_cls"]=="clf_lgbm") 
        ]["raw.sMAPE"].describe()
r_now = results[
    (results["c_imputer"]=="imputer_dummy") &
    (results["c_model"]=="even_weighting") &
    (results["c_fintransformer"]=="ft_scaling_power") &
    (results["c_preprocessor"]=="preprocessor_normed_iid") &
    (results["c_bucket_cls"]=="clf_lgbm") 
        ]["raw.sMAPE"].describe()
r_worst = results[
    (results["c_imputer"]=="imputer_revenue_bucket_10") &
    (results["c_model"]=="default_weighting") &
    (results["c_fintransformer"]=="ft_scaling_robust") &
    (results["c_preprocessor"]=="preprocessor_baseline") &
    (results["c_bucket_cls"]=="clf_rf") 
        ]["raw.sMAPE"].describe()

r_best = results[
    (results["c_imputer"]=="imputer_dummy") &
    (results["c_model"]=="default_weighting") &
    (results["c_fintransformer"]=="ft_scaling_power") &
    (results["c_preprocessor"]=="preprocessor_iid") &
    (results["c_bucket_cls"]=="clf_rf") 
        ]["raw.sMAPE"].describe()

r_average = results["raw.sMAPE"].describe()


pd.DataFrame([r_old,r_now, r_worst, r_best, r_average], index="old,now,worst,best,all".split(",")).style.highlight_min(color = 'blue',  
                       axis = 0).highlight_max(color = 'darkred',  
                       axis = 0)
# %%
