from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing_extensions import Self

from base import OxariConfidenceEstimator, OxariScopeEstimator
from base.oxari_types import ArrayLike
from scipy import spatial

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
        self.confidence_estimator = MapieRegressor(estimator=self.pipeline.estimator).fit(X, y)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.pipeline._preprocess(X)
        preds = self.pipeline.estimator.predict(X)
        residuals_pred = self.confidence_estimator.predict(X, alpha=.05)[1].reshape(-1,2)
        bottom = residuals_pred[:, 0]
        top = residuals_pred[:, 1]
        df = self._construct_result(top, bottom, preds)
        return df

# TODO: Idea for conformal predictions based on similar instances
# 1. Find a group of size K of nearest neighbors
# 2. Predict conformity scores |y-y_pred| for all neighbors
# 3. Quantile order by the conformity score
# 4. Compute the 1-eps quantile of these neighbors alpha
# 5. Assign boundary based on the chosen eps -> y_pred_new +/- alpha
# 

class ConformalKNNConfidenceEstimator(OxariConfidenceEstimator):
    """
    From here: https://medium.com/towards-data-science/you-cant-predict-the-errors-of-your-model-or-can-you-1a2e4a1f38a0

    """

    def __init__(self, k=10, **kwargs) -> None:
        super().__init__(**kwargs)
        self._all_estimators: List[OxariScopeEstimator] = []
        self.res = []
        self.k = k
        self.confidence_estimator:MapieRegressor = None

    def fit(self, X, y, **kwargs) -> Self:
        X = self.pipeline._preprocess(X)
        y = self.pipeline._transform_scope(y)
        self.grouper = spatial.KDTree(X)
        # q_estimations = self.pipeline.estimator.predict(X)
        y_pred = self.pipeline.estimator.fit(X, y)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        X = self.pipeline._preprocess(X)
        # https://stackoverflow.com/a/32446753/4162265
        k_closest_idx = self.grouper.query(X, self.k)[1]
        k_closest = self.grouper.data[k_closest_idx]
        num_sample, num_neighbors, num_ft = k_closest.shape
        k_closest_reshaped = pd.DataFrame(k_closest.reshape(-1, num_ft), columns=X.columns)
        y_pred = self.pipeline.estimator.predict(k_closest_reshaped).reshape(num_sample, num_neighbors)
        # y_pred_sorted = np.sort(y_pred, axis=1)
        quantiles = pd.DataFrame(np.quantile(y_pred, q=[0.05, 0.5 ,0.95], axis=1).T, columns=["bottom","median","top"])
        bottom = quantiles["bottom"]
        top = quantiles["top"]
        preds = quantiles["median"]
        df = self._construct_result(top, bottom, preds)
        return df

# TODO: Implement conformalized quantile regression
# 1. https://github.com/yromano/cqr/blob/master/cqr_real_data_example.ipynb
# 2. https://towardsdatascience.com/how-to-predict-risk-proportional-intervals-with-conformal-quantile-regression-175775840dc4   