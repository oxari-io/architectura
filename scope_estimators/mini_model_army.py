from typing import Union
from base import OxariScopeEstimator, DefaultRegressorEvaluator
from base.helper import BucketScopeDiscretizer
import numpy as np
import pandas as pd
from scope_estimators.mma.classifier import BucketClassifier, ClassifierOptimizer, BucketClassifierEvauator
from scope_estimators.mma.regressor import BucketRegressor, RegressorOptimizer
from base.oxari_types import ArrayLike

N_TRIALS = 1
N_STARTUP_TRIALS = 1


class MiniModelArmyEstimator(OxariScopeEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(**kwargs)
        self.n_buckets = n_buckets
        self.discretizer = BucketScopeDiscretizer(self.n_buckets)
        self.bucket_cl: BucketClassifier = BucketClassifier().set_optimizer(ClassifierOptimizer(n_trials=self.n_trials,
                                                                                                n_startup_trials=self.n_startup_trials)).set_evaluator(BucketClassifierEvauator())
        self.bucket_rg: BucketRegressor = BucketRegressor().set_optimizer(RegressorOptimizer(n_trials=self.n_trials,
                                                                                             n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        y_binned = self.discretizer.fit_transform(X, y)
        self.bucket_cl: BucketClassifier = self.bucket_cl.set_params(**self.params.get("cls", {})).fit(X, y_binned)
        groups = self.bucket_cl.predict(X)
        self.bucket_rg = self.bucket_rg.set_params(**self.params.get("rgs", {})).fit(X, y, groups=groups)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        y_binned_pred = self.bucket_cl.predict(X)
        y_pred = self.bucket_rg.predict(X, groups=y_binned_pred)
        return y_pred

    # def set_params(self, **params):
    #     # self.cls=params.pop("cls", {})
    #     # self.rgs=params.pop("rgs", {})
    #     return super().set_params(**params)

    def get_config(self, deep=True):
        result = {"n_buckets": self.n_buckets, **super().get_config(deep)}
        if deep:
            result = {**result, "cls": self.bucket_cl.get_config(), "rgs": self.bucket_rg.params}
        return {**result}

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        self.discretizer = self.discretizer.fit(X_train, y_train)
        y_train_binned = self.discretizer.transform(y_train)
        y_val_binned = self.discretizer.transform(y_val)

        # TODO: Maybe they need to be connected
        best_params_cls, info_cls = self.bucket_cl.optimize(X_train, y_train_binned, X_val, y_val_binned, **kwargs)
        best_params_rgs, info_rgs = self.bucket_rg.optimize(X_train, y_train, X_val, y_val, grp_train=y_train_binned, grp_val=y_val_binned, **kwargs)

        # return {**best_params_cls,**best_params_rgs}, {"classifier":info_cls, "regressor":info_rgs}
        return {"cls": best_params_cls, "rgs": best_params_rgs}, {"classifier": info_cls, "regressor": info_rgs}

    def evaluate(self, y_true, y_pred, **kwargs):
        X_test = kwargs.get("X_test")
        y_true_bins = self.discretizer.transform(y_true) 

        y_pred_cl = self.bucket_cl.predict(X_test)
        results_cl = self.bucket_cl.evaluate(y_true_bins, y_pred_cl)

        y_pred_rg = self.bucket_rg.predict(X_test, groups=y_true_bins)
        results_rg = self.bucket_rg.evaluate(y_true, y_pred_rg)

        results_end_to_end = self._evaluator.evaluate(y_true, y_pred)
        combined_results = {"n_buckets": self.n_buckets, "classifier": results_cl, "regressor": results_rg, **results_end_to_end}
        return combined_results


class MiniModelArmyClusterBucketEstimator(MiniModelArmyEstimator):
    # TODO: instead of classifier uses clustering for bucketing
    pass