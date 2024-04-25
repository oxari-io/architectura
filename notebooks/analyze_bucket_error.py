# %%
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects
import numpy as np

# %%
cwd = pathlib.Path(__file__).parent
results_rgs = pd.read_csv(cwd.parent / 'local/eval_results/experiment_bucket_error_regression.csv')
# Hacking in forgotten reported value
results_rgs
# %%
cwd = pathlib.Path(__file__).parent
results_cls = pd.read_csv(cwd.parent / 'local/eval_results/experiment_bucket_error_classification.csv')
# Hacking in forgotten reported value
results_cls.index = ["precision", "recall", "f1", "support"]
results_cls.T

# %%
