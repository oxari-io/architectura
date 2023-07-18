from base.common import OxariScopeEstimator, OxariClassifier
from base.oxari_types import ArrayLike
from scope_estimators import MiniModelArmyEstimator
from typing_extensions import Self
import pandas as pd

from scope_estimators.mma.classifier import BucketClassifier

class SectorModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_trials=1, n_startup_trials=1, **kwargs):
        super().__init__(n_trials, n_startup_trials, **kwargs)
        self.bucket_cl: SectorClassifier = SectorClassifier()

    def fit(self, X, y, **kwargs) -> Self:
        # NOTE: Question is whether the linkage between bucket_cl prediction and regression makes sense. I start to believe it does not. 
        # If the classfier predicts one class only the regressor will just use the full data.
        # If the classifier predicts the majority class the model will have one powerful bucket and others are weak.
        # Seperate learning allows to every bucket to learn according to the data distribution. The error does not propagate.  
        self.n_features_in_ = X.shape[1]
        y_binned =  X[self.bucket_cl.sector_column]
        self.bucket_cl: SectorClassifier = self.bucket_cl.set_params(**self.params.get("cls", {})).fit(X, y_binned)
        groups = self.bucket_cl.predict(X)
        self.bucket_rg = self.bucket_rg.set_params(**self.params.get("rgs", {})).fit(X, y_binned, groups=groups)
        # self.bucket_rg = self.bucket_rg.set_params(**self.params.get("rgs", {})).fit(X, y, groups=groups)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        y_binned_pred = self.bucket_cl.predict(X)
        y_pred = self.bucket_rg.predict(X, groups=y_binned_pred)
        return y_pred

    # def set_params(self, **params):
    #     # self.cls=params.pop("cls", {})
    #     # self.rgs=params.pop("rgs", {})
    #     return super().set_params(**params)

    # def get_config(self, deep=True):
    #     result = {"n_buckets": self.n_buckets, **super().get_config(deep)}
    #     if deep:
    #         result = {**result, "cls": self.bucket_cl.get_config(), "rgs": self.bucket_rg.params}
    #     return {**result}

    def optimize(self, X_train, y_train, X_val, y_val, **kwargs):
        # TODO: Maybe they need to be connected
        # best_params_cls, info_cls = self.bucket_cl.optimize(X_train, y_train_binned, X_val, y_val_binned, **kwargs)
        best_params_cls = {}
        info_cls = pd.DataFrame()
        y_train_binned = self.bucket_cl.predict(X_train)
        y_val_binned = self.bucket_cl.predict(X_val)
    
        best_params_rgs, info_rgs = self.bucket_rg.optimize(X_train, y_train, X_val, y_val, grp_train=y_train_binned, grp_val=y_val_binned, **kwargs)

        # return {**best_params_cls,**best_params_rgs}, {"classifier":info_cls, "regressor":info_rgs}
        return {"cls": best_params_cls, "rgs": best_params_rgs}, {"classifier": info_cls, "regressor": info_rgs}

    def evaluate(self, y_true, y_pred, **kwargs):
        X_test = kwargs.get("X_test")

        y_pred_cl = self.bucket_cl.predict(X_test)
        # results_cl = self.bucket_cl.evaluate(y_true_bins, y_pred_cl)

        y_pred_rg = self.bucket_rg.predict(X_test, groups=y_pred_cl)
        results_rg = self.bucket_rg.evaluate(y_true, y_pred_rg)

        results_end_to_end = self._evaluator.evaluate(y_true, y_pred)
        combined_results = {"n_buckets": self.n_buckets, "regressor": results_rg, **results_end_to_end}
        return combined_results
    
class SectorClassifier(BucketClassifier):
    def __init__(self, sector_column='ft_catm_sector_name', **kwargs):
        super().__init__(**kwargs)
        self.sector_column = sector_column

