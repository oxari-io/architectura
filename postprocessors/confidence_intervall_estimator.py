from base import OxariPostprocessor, OxariScopeEstimator, OxariDataManager, OxariMetaModel, OxariRegressor, OxariTransformer, OxariPipeline
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Union, List
from base.oxari_types import ArrayLike
from sklearn.base import MultiOutputMixin
from sklearn.model_selection import KFold


class OxariConfidenceEstimator(OxariRegressor, MultiOutputMixin):
    def __init__(self, object_filename=None, **kwargs) -> None:
        super().__init__(object_filename, **kwargs)


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
    def __init__(self, object_filename=None, estimator: OxariPipeline = None, **kwargs) -> None:
        super().__init__(object_filename, **kwargs)
        self.estimator = estimator

    def fit(self, X, y, **kwargs) -> "OxariRegressor":
        y_hat = self.estimator.predict(X, **kwargs)
        residuals = np.abs(y_hat - y)
        self.error_range = np.quantile(residuals, q=95)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs) -> ArrayLike:
        df = pd.DataFrame()
        df['pred'] = self.estimator.predict(X, **kwargs)
        df['upper'] = df['pred'] + self.error_range
        df['lower'] = df['pred'] - self.error_range

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
    def __init__(self, object_filename=None, model: OxariScopeEstimator = None, **kwargs) -> None:
        super().__init__(object_filename, **kwargs)
        self.estimator = model
        self._estimators: List[OxariScopeEstimator] = []
        self.res = []
        self.alpha = 0.95

    def fit(self, X, y, **kwargs) -> "OxariRegressor":
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train_, X_test_ = X[train_index], X[test_index]
            y_train_, y_test_ = y[train_index], y[test_index]

            self.estimator.fit(X_train_, y_train_)
            self._estimators.append(self.estimator)
            self.res.extend(list(y_test_ - self.estimator.predict(X_test_)))

    def predict(self, X, **kwargs) -> ArrayLike:
        y_pred_multi = np.column_stack([e.predict(X) for e in self._estimators])
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

        preds = np.median(y_pred_multi, axis=1)
        df = pd.DataFrame()
        df['pred'] = preds
        df['upper'] = top
        df['lower'] = bottom
        return df
