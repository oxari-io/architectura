# pip install autoimpute
import time
from IPython.core.pylabtools import figsize
from lightgbm import LGBMRegressor

import pandas as pd
from base import BaselineConfidenceEstimator, OxariDataManager, OxariImputer
from base.dataset_loader import CompanyDataFilter, SimpleDataFilter
from base.helper import LogTargetScaler
from base.run_utils import get_default_datamanager_configuration, get_remote_datamanager_configuration, get_small_datamanager_configuration
from feature_reducers import PCAFeatureReducer
from imputers import RevenueQuantileBucketImputer, KMeansBucketImputer, KMedianBucketImputer, BaselineImputer, RevenueBucketImputer, AutoImputer, OldOxariImputer, MVEImputer
from datasources import S3Datasource
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import tqdm
import itertools as it

from imputers.categorical import CategoricalStatisticsImputer
from imputers.core import DummyImputer
from imputers.equilibrium_method import EquilibriumImputer
from imputers.interpolation import LinearInterpolationImputer, SplineInterpolationImputer
from imputers.revenue_bucket import RevenueExponentialBucketImputer, RevenueParabolaBucketImputer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":

    all_results = []
    # loads the data just like CSVDataLoader, but a selection of the data
    # TODO: Redesign imputation to be at the start everytime.
    # NOTE: I hereby define vertical interpolation and horizontal interpolation.
    # - Vertical interpolation interpolates the NA's the column independently of other columns. Usually grouped by company.
    # - Horizontal interpolation does not take any other row into account for imputation. Basically making it time-independent.
    difficulties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dataset = get_small_datamanager_configuration(0.1).run()

    bag = dataset.get_split_data(OxariDataManager.ORIGINAL)
    SPLIT_1 = bag.scope_1
    X, Y = SPLIT_1.train
    X_new = X.copy()
    # X_new[X.filter(regex='^ft_num', axis=1).columns] = minmax_scale(X.filter(regex='^ft_num', axis=1))

    X_train, X_test = train_test_split(X_new, test_size=0.5)
    keeping_criterion_2 = (X_test.isna().mean(axis=0)<0.5)
    keep_columns_2 = X_train.loc[:, keeping_criterion_2].columns

    imputer_2: EquilibriumImputer = EquilibriumImputer(max_iter=100).clone()
    X_subset = X_train[keep_columns_2]
    imputer_2 = imputer_2.fit(X_subset)


    imputer_2.evaluate(X_subset, p=0.1)
    diffs = pd.DataFrame(np.vstack(imputer_2.history_diffs), columns=imputer_2._features_transformed)
    mimss = pd.DataFrame(np.vstack(imputer_2.history_mims), columns=imputer_2._features_transformed)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.lineplot(data=diffs.reset_index().melt('index'), x='index', y='value', hue='variable', ax=axes[0])
    sns.lineplot(data=mimss.reset_index().melt('index'), x='index', y='value', hue='variable', ax=axes[1])
    fig.tight_layout()
    plt.show()










