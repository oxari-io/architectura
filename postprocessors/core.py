import pandas as pd
import numpy as np
from base import OxariScopeEstimator, OxariPipeline
import shap
from xgboost import XGBRegressor, XGBClassifier, XGBModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from base.metrics import smape
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from xgboost import plot_tree
import abc
from base.oxari_types import ArrayLike

def get_jump_rate(y_pre, y_post):
    result = np.maximum(y_pre, y_post) / np.minimum(y_pre, y_post)
    return result


def custom_masker(mask, x):
    x_ = x.copy()
    x_[~mask] = None

    return x_.reshape(1, len(x))  # in this simple example we just zero out the features we are masking
    # return pd.Series(zip(features, x_)).to_frame() # in this simple example we just zero out the features we are masking


class OxariExplainer(abc.ABC):
    BINARIES_PREFIX = 'categoricals'

    def _create_canvas(self, fig, ax):
        if fig and ax:
            # Both are provided => Nothing needs to be created
            pass

        if (not fig) and (not ax):
            # Nothing is provided => Both are needed.
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot()

        if (fig) and (not ax):
            # Only figure is provided => Ax needs to be created.
            ax = fig.add_subplot()

        if (not fig) and ax:
            # Only ax is proided => No need for a figure. Plot on ax.
            pass

        return fig, ax

    @abc.abstractmethod
    def visualize(self):
        pass

def run_shap(estimator):
    explainer = shap.Explainer(estimator.predict, custom_masker)
    pass

# class UselessFeaturesExplainer(OxariExplainer):
#     def __init__(self, pipeline: OxariPipeline, sample_size=100, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.pipeline = pipeline
#         self.sample_size = sample_size
#         self.rfe = RFECV(self.pipeline.estimator, importance_getter=run_shap)
        
#     def fit(self, X, y, **kwargs) -> Self:
#         X_ = self.pipeline._preprocess(X)
#         y_ = self.pipeline._transform_scope(y)
#         self.rfe.fit(X_, y_, **kwargs)
#         return self

#     def explain(self, X, y, **kwargs) -> Self:
#         self.feature_importances_ = self.rfe.ranking_
#         return Self
    
#     def visualize(self):
#         return super().visualize()


class ShapExplainer(OxariExplainer):

    def __init__(self, estimator: OxariPipeline, sample_size=100, **kwargs) -> None:
        super().__init__(**kwargs)
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

    def visualize(self):
        shap.summary_plot(self.shap_values, self.X, show=False)
        # TODO: Should also return fig and ax like the other explainers.
        fig = plt.gcf()
        ax = plt.gca()
        return fig, ax


class SurrogateExplainerMixin(abc.ABC):
    BINARIES_PREFIX = "categorical"

    def __init__(self, estimator:OxariPipeline, surrogate:XGBModel, **kwargs) -> None:
        super().__init__()
        self.estimator = estimator
        self.surrogate_model: XGBModel = surrogate
        self.pipeline: Pipeline = None
        self.name = self.__class__.__name__

#  TODO: Needs function to evalute how faithfull the surrogate is
#  TODO: Needs function to optimize for most faithfull surrogacy
class TreeBasedExplainerMixin(OxariExplainer, SurrogateExplainerMixin, abc.ABC):

    def __init__(self, estimator: OxariPipeline, surrogate, topk_features=20, **kwargs) -> None:
        super().__init__(estimator=estimator, surrogate=surrogate ,**kwargs)
        self.feature_importances_: ArrayLike = None
        self.topk_features = topk_features
        # self._init_preprocessor()
        # self._init_surrogate_model()

    def _init_feature_names(self):
        self.feature_names_out_ = list(self.pipeline[:-1].get_feature_names_out())
        self.pipeline[-1].get_booster().feature_names = self.feature_names_out_

    def _init_preprocessor(self, df:pd.DataFrame):
        cat_cols = df.columns[df.columns.str.startswith('ft_cat')]
        self.cat_transform = ColumnTransformer([(self.BINARIES_PREFIX, OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist'), cat_cols)],
                                               remainder='passthrough')

    def _init_surrogate_model(self):
        self.pipeline = Pipeline([('cat-encode', self.cat_transform), ('regress', self.surrogate_model)])

    def plot_importances(self, fig=None, ax=None):
        fig, ax = self._create_canvas(fig, ax)
        tmp = pd.DataFrame()
        tmp['feature'] = self.feature_names_out_
        tmp['importance'] = self.feature_importances_
        tmp_sorted_subset = tmp.sort_values('importance', ascending=False).iloc[:self.topk_features]
        ax = sns.barplot(data=tmp_sorted_subset, y='feature', x='importance', ax=ax)
        for (index, row), p in zip(tmp_sorted_subset.iterrows(), ax.patches):
            ax.text(x=p.get_x(), y=p.get_y() + p.get_height() / 2, s=row["feature"], color='black', ha="left", va='center')
        ax.get_yaxis().set_ticks([])
        ax.set_title(self.name)
        return fig, ax

    def plot_tree(self, fig=None, ax=None):
        fig, ax = self._create_canvas(fig, ax)
        ax = plot_tree(self.pipeline[-1], ax=ax, rankdir='LR')
        ax.set_title(self.name)
        return fig, ax

    def visualize(self):
        fig, axes = plt.subplots(2, 1, figsize=(15, 7))
        ax1, ax2 = axes
        fig, ax1 = self.plot_importances(fig, ax1)
        fig, ax2 = self.plot_tree(fig, ax2)
        fig.suptitle(f'{self.name}')
        fig.tight_layout()
        return fig, axes


class ResidualExplainer(TreeBasedExplainerMixin):
    
    def __init__(self, estimator: OxariPipeline, **kwargs) -> None:
        super().__init__(estimator=estimator, surrogate=XGBRegressor(), **kwargs)

    def fit(self, X, y, **kwargs):
        self._init_preprocessor(X)
        self._init_surrogate_model()        
        y_hat = self.estimator.predict(X)
        self.pipeline.fit(X, y - y_hat, **kwargs)
        self._init_feature_names()
        return self

    def explain(self, X, y, **kwargs):
        y_true = y - self.estimator.predict(X)
        y_hat = self.pipeline.predict(X)
        # TODO: Inherit from OxariMixin... Consider that shap explainer istn't evaluatable
        # TODO: Use defaultregression evaluator and defaultclassification evaluator here!
        self.unfaithfulness = smape(y_true, y_hat)
        self.feature_importances_ = self.surrogate_model.feature_importances_
        return self


class JumpRateExplainer(TreeBasedExplainerMixin):

    def __init__(self, estimator: OxariPipeline, threshhold=1.2, **kwargs) -> None:
        super().__init__(estimator=estimator, surrogate=XGBClassifier(), **kwargs)
        self.threshold = threshhold

    def fit(self, X, y, **kwargs):
        self._init_preprocessor(X)
        self._init_surrogate_model()         
        y_hat = self.estimator.predict(X)
        y_jump_rate = get_jump_rate(y, y_hat)
        y_target = y_jump_rate < self.threshold
        self.pipeline.fit(X, y_target, **kwargs)
        self._init_feature_names()
        return self

    def explain(self, X, y, **kwargs):
        y_true = y - self.estimator.predict(X)
        y_jump_rate = get_jump_rate(y, y_true)
        y_target = y_jump_rate < self.threshold
        y_hat = self.pipeline.predict(X)
        self.unfaithfulness = classification_report(y_target, y_hat)
        self.feature_importances_ = self.surrogate_model.feature_importances_
        return self


class DecisionExplainer(TreeBasedExplainerMixin):

    def __init__(self, estimator: OxariScopeEstimator, threshhold=1.2, **kwargs) -> None:
        super().__init__(estimator=estimator, surrogate=XGBRegressor(), **kwargs)
        self.threshold = threshhold

    def fit(self, X, y, **kwargs):
        self._init_preprocessor(X)
        self._init_surrogate_model()         
        y_hat = self.estimator.predict(X)
        self.pipeline.fit(X, y_hat, **kwargs)
        self._init_feature_names()
        return self

    def explain(self, X, y, **kwargs):
        y_true = self.estimator.predict(X)
        y_hat = self.pipeline.predict(X)
        self.unfaithfulness = smape(y_true, y_hat)
        self.feature_importances_ = self.surrogate_model.feature_importances_
        return self
    
    
