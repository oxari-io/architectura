from base import OxariScopeEstimator
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List
from base.oxari_types import ArrayLike
from sklearn.model_selection import KFold
from base import OxariConfidenceEstimator
from typing_extensions import Self
import lightgbm as lgb
from mapie.regression import MapieRegressor


# TODO: Implement evaluator that computes the coverage. https://towardsdatascience.com/prediction-intervals-in-python-64b992317b1a
class BaselineConfidenceEstimator(OxariConfidenceEstimator):
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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs) -> Self:
        X = self.pipeline._preprocess(X)
        y = self.pipeline._transform_scope(y)
        y_hat = self.pipeline.estimator.predict(X, **kwargs)
        residuals = pd.DataFrame(np.abs(y_hat - y)).dropna()
        self.error_range = np.quantile(residuals, q=1 - self.alpha)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.pipeline._preprocess(X)
        mean_ = self.pipeline.estimator.predict(X, **kwargs)
        df = pd.DataFrame()
        bottom = mean_ - self.error_range
        preds = mean_
        top = mean_ + self.error_range
        df = self._construct_result(top, bottom, preds)

        return df


class PercentileOffsetConfidenceEstimator(OxariConfidenceEstimator):
    """
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs) -> Self:
        X = self.pipeline._preprocess(X)
        y = self.pipeline._transform_scope(y)
        y_hat = self.pipeline.estimator.predict(X, **kwargs)
        offsets = np.maximum(y_hat, y) / np.minimum(y_hat, y)
        self.error_range = np.quantile(offsets, q=1 - self.alpha)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.pipeline._preprocess(X)
        mean_ = self.pipeline.estimator.predict(X, **kwargs)
        bottom = mean_ / self.error_range
        preds = mean_
        top = mean_ * self.error_range
        df = self._construct_result(top, bottom, preds)

        return df


class ProbablisticConfidenceEstimator(OxariConfidenceEstimator):
    """
    For Probablistic models that already have a native way to predict the standard deviation.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs) -> Self:
        X = self.pipeline._preprocess(X)
        y = self.pipeline._transform_scope(y)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.pipeline._preprocess(X)
        mean_, std_ = self.pipeline.predict(X, return_std=True)
        df = pd.DataFrame()
        bottom = mean_ - std_
        preds = mean_
        top = mean_ + std_

        df = self._construct_result(top, bottom, preds)

        return df


class JacknifeConfidenceEstimator(OxariConfidenceEstimator):
    """
    From here: https://towardsdatascience.com/generating-confidence-intervals-for-regression-models-2dd60026fbce
    The method follows the Jacknife+ approach but with KFold CV. 
    The idea is to fit multiple estimators and compute their residuals on unseen data. 
    Then compute the confidence intervall and by taking the alpha quantile of the residuals. 
    Then sum and substract them from the predictions on the real prediction.

    The paper simulations show that this method is a little worse than the Jackknife+, however, it is way faster. 
    In practical terms, this will probably be the method being used in most cases.
    """

    def __init__(self, n_splits=10, **kwargs) -> None:
        super().__init__(**kwargs)
        self._all_estimators: List[OxariScopeEstimator] = []
        self.res = []
        self.n_splits = n_splits

    def fit(self, X, y, **kwargs) -> Self:
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        X = self.pipeline._preprocess(X)
        y = self.pipeline._transform_scope(y)
        for idx, (train_index, test_index) in tqdm(enumerate(kf.split(X)), total=self.n_splits, desc="Jacknife++ Training"):
            X_train_, X_test_ = X.iloc[train_index], X.iloc[test_index]
            y_train_, y_test_ = y.iloc[train_index], y.iloc[test_index]
            new_estimator: OxariScopeEstimator = self.pipeline.estimator.clone()
            new_estimator.fit(X_train_, y_train_)
            self._all_estimators.append(new_estimator)
            self.res.extend(list(y_test_ - new_estimator.predict(X_test_)))
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.pipeline._preprocess(X)
        y_pred_multi = np.column_stack([e.predict(X) for e in self._all_estimators])
        ci = np.quantile(self.res, 1 - self.alpha)
        top = []
        bottom = []
        for i in range(y_pred_multi.shape[0]):
            if ci > 0:
                top.append(np.quantile(y_pred_multi[i] + ci, 1 - self.alpha))
                bottom.append(np.quantile(y_pred_multi[i] - ci, 1 - self.alpha))
            else:
                top.append(np.quantile(y_pred_multi[i] - ci, 1 - self.alpha))
                bottom.append(np.quantile(y_pred_multi[i] + ci, 1 - self.alpha))

        # preds = np.median(y_pred_multi, axis=1)
        preds = self.pipeline.estimator.predict(X)
        df = self._construct_result(top, bottom, preds)
        return df


class DirectLossConfidenceEstimator(OxariConfidenceEstimator):
    """
    From here: https://medium.com/towards-data-science/you-cant-predict-the-errors-of-your-model-or-can-you-1a2e4a1f38a0
    
    """

    def __init__(self, n_splits=10, **kwargs) -> None:
        super().__init__(**kwargs)
        self._all_estimators: List[OxariScopeEstimator] = []
        self.res = []
        self.n_splits = n_splits
        self.confidence_estimator = lgb.LGBMRegressor()

    def fit(self, X, y, **kwargs) -> Self:
        X = self.pipeline._preprocess(X)
        y = self.pipeline._transform_scope(y)
        y_pred = self.pipeline.estimator.predict(X)
        y_residuals = np.abs(y - y_pred)
        self.confidence_estimator = self.confidence_estimator.fit(X, y_residuals)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.pipeline._preprocess(X)
        preds = self.pipeline.estimator.predict(X)
        residuals_pred = self.confidence_estimator.predict(X)
        bottom = preds - residuals_pred
        top = preds + residuals_pred
        df = self._construct_result(top, bottom, preds)
        return df

class MAPIEConfidenceEstimator(OxariConfidenceEstimator):
    """
    From here: https://medium.com/towards-data-science/you-cant-predict-the-errors-of-your-model-or-can-you-1a2e4a1f38a0

    """

    def __init__(self, n_splits=10, **kwargs) -> None:
        super().__init__(**kwargs)
        self._all_estimators: List[OxariScopeEstimator] = []
        self.res = []
        self.n_splits = n_splits
        self.confidence_estimator:MapieRegressor = None

    def fit(self, X, y, **kwargs) -> Self:
        X = self.pipeline._preprocess(X)
        y = self.pipeline._transform_scope(y)
        self.confidence_estimator = MapieRegressor(estimator=self.pipeline.estimator, cv='prefit').fit(X, y)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.pipeline._preprocess(X)
        preds = self.pipeline.estimator.predict(X)
        residuals_pred = self.confidence_estimator.predict(X, alpha=.05)[1].reshape(-1,2)
        bottom = residuals_pred[:, 0]
        top = residuals_pred[:, 1]
        df = self._construct_result(top, bottom, preds)
        return df
