import itertools
import pandas as pd
import numpy as np
from shap import KernelExplainer, Explainer
from base import OxariMetaModel, OxariScopeEstimator
import shap
# from shap.explainers import maskers
from base.metrics import smape
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from base.mappings import CatMapping, NumMapping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import seaborn as sns
from base.metrics import smape
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from xgboost import plot_tree
import io
import tempfile


def custom_masker(mask, x):
    x_ = x.copy()
    x_[~mask] = None

    return x_.reshape(1, len(x))  # in this simple example we just zero out the features we are masking
    # return pd.Series(zip(features, x_)).to_frame() # in this simple example we just zero out the features we are masking


class ShapExplainer():

    def __init__(self, estimator: OxariScopeEstimator, sample_size=100, **kwargs) -> None:
        self.estimator = estimator
        self.sample_size = sample_size
        shap.initjs()

    def print_accuracy(self, X, y):
        y_hat = self.estimator.predict(X)
        print("Root mean squared test error = {0}".format(smape(y, y_hat)))
        # time.sleep(0.5) # to let the print get out before any progress bars

    def fit(self, X, y, **kwargs):
        # X_train_summary = shap.kmeans(X, 10)
        self.ex = shap.Explainer(self.estimator.predict, custom_masker)
        return self

    def explain(self, X, y, **kwargs):
        self.X = X.sample(self.sample_size)
        self.shap_values = self.ex(self.X)
        return self

    def plot(self):
        shap.summary_plot(self.shap_values, self.X)


class ResidualFeatureImportanceExplainer():

    def __init__(self, estimator: OxariScopeEstimator, topk_features=20, **kwargs) -> None:
        self.estimator = estimator
        self.surrogate_model = XGBRegressor()
        self.topk_features = topk_features
        # self.cat_transform = ColumnTransformer([('categoricals', ce.OneHotEncoder(), CatMapping.get_features())], remainder='passthrough')
        self.cat_transform = ColumnTransformer([('categoricals', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist'), CatMapping.get_features())],
                                               remainder='passthrough')
        self.pipeline = Pipeline([('cat-encode', self.cat_transform), ('regress', self.surrogate_model)])

    def fit(self, X, y, **kwargs):
        y_hat = self.estimator.predict(X)
        self.pipeline.fit(X, y - y_hat, **kwargs)
        self.feature_names_out_ = self.pipeline[:-1].get_feature_names_out()
        return self

    def explain(self, X, y, **kwargs):
        y_true = y - self.estimator.predict(X)
        y_hat = self.pipeline.predict(X)
        self.unfaithfulness = smape(y_true, y_hat)
        self.feature_importances_ = self.surrogate_model.feature_importances_
        return self

    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(15, 7))
        tmp = pd.DataFrame()
        tmp['feature'] = self.feature_names_out_
        tmp['importance'] = self.feature_importances_
        tmp_sorted_subset = tmp.sort_values('importance', ascending=False).iloc[:self.topk_features]

        ax = sns.barplot(data=tmp_sorted_subset, y='feature', x='importance', ax=ax)
        fig.tight_layout()
        return plt.show(block=True)


def get_jump_rate(y_pre, y_post):
    result = np.maximum(y_pre, y_post) / np.minimum(y_pre, y_post)
    return result


class JumpRateExplainer():
    BINARIES_PREFIX = 'categoricals'

    def __init__(self, estimator: OxariScopeEstimator, topk_features=20, threshhold=1.2, **kwargs) -> None:
        self.estimator = estimator
        self.surrogate_model = XGBClassifier()
        self.topk_features = topk_features
        self.threshold = threshhold
        # self.cat_transform = ColumnTransformer([('categoricals', ce.OneHotEncoder(), CatMapping.get_features())], remainder='passthrough')
        self.cat_transform = ColumnTransformer([(self.BINARIES_PREFIX, OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist'), CatMapping.get_features())],
                                               remainder='passthrough')
        self.pipeline = Pipeline([('cat-encode', self.cat_transform), ('regress', self.surrogate_model)])

    def fit(self, X, y, **kwargs):
        y_hat = self.estimator.predict(X)
        y_jump_rate = get_jump_rate(y, y_hat)
        y_target = y_jump_rate < self.threshold
        self.pipeline.fit(X, y_target, **kwargs)
        self.feature_names_out_ = list(self.pipeline[:-1].get_feature_names_out())
        self.surrogate_model.get_booster().feature_names = self.feature_names_out_ 
        return self

    def explain(self, X, y, **kwargs):
        y_true = y - self.estimator.predict(X)
        y_jump_rate = get_jump_rate(y, y_true)
        y_target = y_jump_rate < self.threshold
        y_hat = self.pipeline.predict(X)
        self.unfaithfulness = classification_report(y_target, y_hat)
        self.feature_importances_ = self.surrogate_model.feature_importances_
        return self

    def _create_canvas(self, fig, ax):
        if fig and ax:
            # Both are provided => Nothing needs to be created
            pass

        if (not fig) and (not ax):
            # Nothing is provided => Both are needed.
            fig = plt.figure(figsize=(15, 7))
            ax = fig.add_subplot()

        if (fig) and (not ax):
            # Only figure is provided => Ax needs to be created.
            ax = fig.add_subplot()

        if (not fig) and ax:
            # Only ax is proided => No need for a figure. Plot on ax.
            pass

        return fig, ax

    def plot_importances(self, fig=None, ax=None):
        fig, ax = self._create_canvas(fig, ax)
        tmp = pd.DataFrame()
        tmp['feature'] = self.feature_names_out_
        tmp['importance'] = self.feature_importances_
        tmp_sorted_subset = tmp.sort_values('importance', ascending=False).iloc[:self.topk_features]
        ax = sns.barplot(data=tmp_sorted_subset, y='feature', x='importance', ax=ax)
        for (index, row), p in zip(tmp_sorted_subset.iterrows(), ax.patches):
            ax.text(x=p.get_x(), y=p.get_y()+p.get_height()/2, s=row["feature"], color='black', ha="left", va='center')        
        ax.get_yaxis().set_ticks([])
        return fig, ax

    def plot_tree(self, fig=None, ax=None):
        fig, ax = self._create_canvas(fig, ax)
        ax = plot_tree(self.pipeline[-1], ax=ax, rankdir='LR')
        return fig, ax

    def plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(15, 7))
        ax1, ax2 = axes
        fig, ax1 = self.plot_importances(fig, ax1)
        fig, ax2 = self.plot_tree(fig, ax2)
        fig.tight_layout()
        return fig, axes


class DecisionExplainer():

    def __init__(self, estimator: OxariScopeEstimator, **kwargs) -> None:
        self.estimator = estimator
        self.surrogate_model = XGBRegressor()

        self.cat_transform = ColumnTransformer([('categoricals', OneHotEncoder(drop='first'), CatMapping.get_features())], remainder='passthrough')
        self.pipeline = Pipeline([('cat-encode', self.cat_transform), ('regress', self.surrogate_model)])

    def fit(self, X, y, **kwargs):
        y_hat = self.estimator.predict(X)
        self.pipeline.fit(X, y_hat, **kwargs)
        return self

    def explain(self, X, y, **kwargs):
        y_true = self.estimator.predict(X)
        y_hat = self.pipeline.predict(X)
        self.unfaithfulness = smape(y_true, y_hat)
        self.feature_importances_ = self.surrogate_model.feature_importances_
        return self

    def plot(self):
        sns.barplot(self.feature_importances_)