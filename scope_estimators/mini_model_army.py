from typing import Union
from base import OxariScopeEstimator
import numpy as np
import pandas as pd

from scope_estimators.mma.classifier import BucketClassifier, ClassfierScopeDiscretizer, ClassifierOptimizer
from scope_estimators.mma.regressor import BucketRegressor, RegressorOptimizer
from scope_estimators.mma.evaluators import ClassifierEvaluator, RegressorEvaluator


class MiniModelArmyEstimator(OxariScopeEstimator):
    def __init__(self, n_buckets=5, **kwargs):
        super().__init__(**kwargs)
        self.discretizer = ClassfierScopeDiscretizer(n_buckets)
        self.bucket_cl: BucketClassifier = BucketClassifier(object_filename="CL_scope_1", scope="scope_1").set_optimizer(ClassifierOptimizer()).set_evaluator(ClassifierEvaluator())
        self.bucket_rg: BucketRegressor = BucketRegressor(object_filename="CR_scope_1", scope="scope_1").set_optimizer(RegressorOptimizer()).set_evaluator(RegressorEvaluator())

    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        y_binned = self.discretizer.transform(y)
        self.bucket_cl:BucketClassifier = self.bucket_cl.fit(X, y_binned, **kwargs.get("cls"))
        groups = self.bucket_cl.predict(X)
        self.bucket_rg = self.bucket_rg.fit(X, y, groups=groups, **kwargs.get("rgs"))
        return self

    def predict(self, X, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        y_binned_pred = self.bucket_cl.predict(X)
        y_pred = self.bucket_rg.predict(X, groups=y_binned_pred)
        return y_pred

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
        y_pred_bins = self.bucket_cl.predict(X_test)
        
        results_cl = self.bucket_cl.evaluate(y_true_bins, y_pred_bins)
        results_rg = self.bucket_rg.evaluate(y_true, y_pred, groups=y_pred_bins)
        combined_results = {**results_cl, **results_rg}
        return combined_results

    def check_conformance(self):
        pass

    def deploy(self):
        pass

    def run(self):
        pass