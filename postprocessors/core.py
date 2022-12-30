import itertools
import pandas as pd
import numpy as np
from shap import KernelExplainer, Explainer
from base import OxariMetaModel, OxariScopeEstimator
import shap
# from shap.explainers import maskers
from base.metrics import smape
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from base.mappings import CatMapping, NumMapping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import seaborn as sns
from base.metrics import smape
import category_encoders as ce
import matplotlib.pyplot as plt

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


class ResidualExplainer():

    def __init__(self, estimator: OxariScopeEstimator, topk_features=20, **kwargs) -> None:
        self.estimator = estimator
        self.surrogate_model = XGBRegressor()
        self.topk_features = topk_features
        # self.cat_transform = ColumnTransformer([('categoricals', ce.OneHotEncoder(), CatMapping.get_features())], remainder='passthrough')
        self.cat_transform = ColumnTransformer([('categoricals', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist'), CatMapping.get_features())], remainder='passthrough')
        self.pipeline = Pipeline([('cat-encode', self.cat_transform), ('regress', self.surrogate_model)])

    def fit(self, X, y, **kwargs):
        y_hat = self.estimator.predict(X)
        self.pipeline.fit(X, y - y_hat, **kwargs)
        self.feature_names_out_ = self.pipeline[:-1].get_feature_names_out()
        return self

    def explain(self, X, y, **kwargs):
        y_true = y-self.estimator.predict(X)
        y_hat = self.pipeline.predict(X)
        self.unfaithfulness = smape(y_true, y_hat)
        self.feature_importances_ = self.surrogate_model.feature_importances_
        return self

    def plot(self): 
        fig, ax =plt.subplots(1,1, figsize=(15, 7))     
        tmp = pd.DataFrame()
        tmp['feature'] = self.feature_names_out_
        tmp['importance'] = self.feature_importances_
        tmp_sorted_subset = tmp.sort_values('importance', ascending=False).iloc[:self.topk_features]
        
        ax = sns.barplot(data = tmp_sorted_subset, y='feature',x='importance', ax=ax)
        fig.tight_layout()
        return plt.show(block=True)
        
        
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