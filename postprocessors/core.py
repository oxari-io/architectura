import itertools
import pandas as pd 
import numpy as np 
from shap import KernelExplainer, Explainer 
from base import OxariMetaModel, OxariScopeEstimator
import shap
# from shap.explainers import maskers
from base.metrics import smape 

def custom_masker(mask, x):
    x_ = x.copy()
    x_[~mask]=None
    
    return x_.reshape(1,len(x)) # in this simple example we just zero out the features we are masking
    # return pd.Series(zip(features, x_)).to_frame() # in this simple example we just zero out the features we are masking


class ShapExplainer():
    def __init__(self, estimator:OxariScopeEstimator) -> None:
        self.estimator = estimator
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
        self.X = X
        self.shap_values = self.ex(X)
        return self

    def plot(self):
        shap.summary_plot(self.shap_values, self.X)
