from base import OxariPostprocessor, OxariScopeEstimator, OxariDataManager, OxariMetaModel, OxariRegressor, OxariTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Union
from base.oxari_types import ArrayLike

class BaselineConfidenceEstimator(OxariRegressor):
    """
    From here: https://towardsdatascience.com/generating-confidence-intervals-for-regression-models-2dd60026fbce
    The naive method may be the first thing that comes to mind when we are trying to generate confidence intervals. The idea is to use the residuals of our model to estimate how much deviation we can expect from new predictions.

    The algorithm goes as follows:
    - Train the model on the training set
    - Calculate the residuals of the predictions on the training set
    - Select the (1 — alpha) quantile of the distribution of the residuals
    - Sum and subtract each prediction from this quantile to get the limits of the confidence interval
    
    One expects that, since the distribution of the residuals is known, the new predictions should not deviate much from it.
    However, this naive solution is problematic because our model may overfit and even if it doesn’t, most of the time the error on the training set will be smaller than the error on the test set, after all, those points are known by the model.
    This may lead to over-optimistic confidence intervals. Therefore, this method should never be used.
    """
    def __init__(self, object_filename=None, model:OxariScopeEstimator=None, **kwargs) -> None:
        super().__init__(object_filename, **kwargs)
        self.estimator = model
    
    def fit(self, X, y, **kwargs) -> "OxariRegressor":
        y_hat = self.estimator.predict(X,y,**kwargs)
        residuals = np.abs(y_hat - y)
        # Most of the error lies in here
        self.error_range = np.quantile(residuals,q=95)
        return super().fit(X, y, **kwargs)
    
    def predict(self, X, **kwargs) -> ArrayLike:
        lb = X - self.error_range
        ub = X + self.error_range
        return np.array(lb, ub)
    
    