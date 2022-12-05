from typing import Union
from base import OxariScopeEstimator, DefaultRegressorEvaluator
import numpy as np
import pandas as pd
from scope_estimators.mma.classifier import BucketClassifier, BucketScopeDiscretizer, ClassifierOptimizer, BucketClassifierEvauator
from scope_estimators.mma.regressor import BucketRegressor, RegressorOptimizer

N_TRIALS = 1
N_STARTUP_TRIALS = 1

class MiniModelArmyEstimator(OxariScopeEstimator):
    def __init__(self, n_buckets=5, **kwargs):
        super().__init__(**kwargs)
        self.n_buckets = n_buckets
        self.discretizer = BucketScopeDiscretizer(self.n_buckets)
        self.bucket_cl: BucketClassifier = BucketClassifier().set_optimizer(ClassifierOptimizer(n_trials=N_TRIALS, num_startup_trials=N_STARTUP_TRIALS)).set_evaluator(BucketClassifierEvauator())
        self.bucket_rg: BucketRegressor = BucketRegressor().set_optimizer(RegressorOptimizer(n_trials=N_TRIALS, n_startup_trials=N_STARTUP_TRIALS)).set_evaluator(DefaultRegressorEvaluator())

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        y_binned = self.discretizer.transform(y)
        self.bucket_cl:BucketClassifier = self.bucket_cl.set_params(**self.cls).fit(X, y_binned)
        groups = self.bucket_cl.predict(X)
        self.bucket_rg = self.bucket_rg.set_params(**self.rgs).fit(X, y, groups=groups)
        return self

    def predict(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        y_binned_pred = self.bucket_cl.predict(X)
        y_pred = self.bucket_rg.predict(X, groups=y_binned_pred)
        return y_pred

    def set_params(self, **params):
        self.cls=params.pop("cls", {})
        self.rgs=params.pop("rgs", {})
        return super().set_params(**params)
    
    def get_params(self, deep=True):
        result = {"n_buckets":self.n_buckets, **super().get_params(deep)}
        if deep:
            result = {**result, **self.bucket_cl.get_params(), **self.bucket_rg.get_params()}
        return result

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        self.discretizer = self.discretizer.fit(y_train)
        y_train_binned = self.discretizer.transform(y_train)
        y_val_binned = self.discretizer.transform(y_val)
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
        combined_results = {"classifier":results_cl, "regressor":results_rg, **results_end_to_end}
        return combined_results

    def check_conformance(self):
        pass

    def run(self):
        pass