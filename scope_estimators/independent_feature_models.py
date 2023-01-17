from typing import Union
from base import OxariScopeEstimator, DefaultRegressorEvaluator, DefaultOptimizer
from base.helper import BucketScopeDiscretizer
import numpy as np
import pandas as pd
from scope_estimators.mma.classifier import BucketClassifier, ClassifierOptimizer, BucketClassifierEvauator
from scope_estimators.mma.regressor import BucketRegressor, RegressorOptimizer
from base.oxari_types import ArrayLike
from sklearn.linear_model import RidgeCV, Ridge

N_TRIALS = 1
N_STARTUP_TRIALS = 1


class IndependentFeatureVotingRegressionEstimator(OxariScopeEstimator):
    """
    This model is a stupid model. Very very stupid. However, it shows what happen if you train a regressor with only one feature on the data.
    The core idea is to train a linear model for each feature. Then take the score weighted average for the prediction.
    We can use this model to test how important interactions are and we can use individual estimators to show how stupid they are.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimators = {}
        self.set_evaluator(DefaultRegressorEvaluator())
        self.set_optimizer(DefaultOptimizer())

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        self.feature_names_in_ = X.columns  # TODO: If pandas dataframe else numeric columns
        self.n_features_in_ = len(self.feature_names_in_)
        for idx, c in enumerate(self.feature_names_in_):
            x = np.array(X)[:, idx, None]
            self._estimators[c] = RidgeCV(**self.params).fit(x, y)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        results = np.zeros((2, self.n_features_in_, len(X)))
        for idx, c in enumerate(self.feature_names_in_):
            x = np.array(X)[:, idx, None]
            results[0, idx] = self._estimators[c].predict(x)
            results[1, idx] = -self._estimators[c].best_score_
        results[1] = 1/results[1]
        results[1] = (results[1] / results[1].sum(axis=0))
        weighted_preds = results[0] * results[1]
        y_pred = weighted_preds.sum(axis=0)
        return y_pred


    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        return self._optimizer.optimize(X_train, y_train, X_val, y_val, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        return self._evaluator.evaluate(y_true, y_pred, **kwargs)
