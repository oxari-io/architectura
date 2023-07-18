from base.common import DefaultClassificationEvaluator, DefaultClusterEvaluator, OxariScopeEstimator, OxariClassifier
from base.oxari_types import ArrayLike
from scope_estimators import MiniModelArmyEstimator
from typing_extensions import Self
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from scope_estimators.mma.classifier import BucketClassifier, ClassifierOptimizer

class SectorModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_trials=1, n_startup_trials=1, cls_optimizer=None, rgs_optimizer=None, **kwargs):
        super().__init__(n_trials, n_startup_trials, **kwargs)
        self.bucket_cl: SectorClassifier = SectorClassifier().set_optimizer(cls_optimizer or ClassifierOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultClassificationEvaluator())
        self.label_encoder: LabelEncoder = LabelEncoder()

    def fit(self, X, y, **kwargs) -> Self:
        # NOTE: Question is whether the linkage between bucket_cl prediction and regression makes sense. I start to believe it does not. 
        # If the classfier predicts one class only the regressor will just use the full data.
        # If the classifier predicts the majority class the model will have one powerful bucket and others are weak.
        # Seperate learning allows to every bucket to learn according to the data distribution. The error does not propagate.  
        self.n_features_in_ = X.shape[1]
        y_binned =  self.label_encoder.fit_transform(X[self.bucket_cl.sector_column])
        self.bucket_cl: SectorClassifier = self.bucket_cl.set_params(**self.params.get("cls", {})).fit(X, y_binned)
        # groups = self.bucket_cl.predict(X)
        self.bucket_rg = self.bucket_rg.set_params(**self.params.get("rgs", {})).fit(X, y, groups=y_binned)
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
        self.label_encoder = self.label_encoder.fit(X_train[self.bucket_cl.sector_column])
        y_train_binned = self.label_encoder.transform(X_train[self.bucket_cl.sector_column]).reshape(-1, 1)
        y_val_binned = self.label_encoder.transform(X_val[self.bucket_cl.sector_column]).reshape(-1, 1)
        
        self.n_buckets = len(self.label_encoder.classes_)

        best_params_cls, info_cls = self.bucket_cl.optimize(X_train, y_train_binned, X_val, y_val_binned, **kwargs)
        best_params_rgs, info_rgs = self.bucket_rg.optimize(X_train, y_train, X_val, y_val, grp_train=y_train_binned, grp_val=y_val_binned, **kwargs)

        # return {**best_params_cls,**best_params_rgs}, {"classifier":info_cls, "regressor":info_rgs}
        return {"cls": best_params_cls, "rgs": best_params_rgs}, {"classifier": info_cls, "regressor": info_rgs}

    def evaluate(self, y_true, y_pred, **kwargs):
        X_test = kwargs.get("X_test")
        y_true_bins = self.label_encoder.transform(X_test[self.bucket_cl.sector_column])

        y_pred_cl = self.bucket_cl.predict(X_test)
        results_cl = self.bucket_cl.evaluate(y_true_bins, y_pred_cl)

        y_pred_rg = self.bucket_rg.predict(X_test, groups=y_true_bins)
        results_rg = self.bucket_rg.evaluate(y_true, y_pred_rg)

        results_end_to_end = self._evaluator.evaluate(y_true, y_pred)
        combined_results = {"n_buckets": self.n_buckets, "classifier": results_cl, "regressor": results_rg, **results_end_to_end}
        return combined_results
    
class SectorClassifier(BucketClassifier):
    def __init__(self, sector_column='ft_catm_sector_name', **kwargs):
        super().__init__(n_buckets = None, **kwargs)
        self.sector_column = sector_column


class DirectSectorModelArmyEstimator(SectorModelArmyEstimator):
    def __init__(self, n_trials=1, n_startup_trials=1, sector_column = 'ft_catm_sector_name', rgs_optimizer=None, **kwargs):
        super().__init__(n_trials, n_startup_trials, **kwargs)
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.sector_column = sector_column

    def fit(self, X, y, **kwargs) -> Self:
        # NOTE: Question is whether the linkage between bucket_cl prediction and regression makes sense. I start to believe it does not. 
        # If the classfier predicts one class only the regressor will just use the full data.
        # If the classifier predicts the majority class the model will have one powerful bucket and others are weak.
        # Seperate learning allows to every bucket to learn according to the data distribution. The error does not propagate.  
        self.n_features_in_ = X.shape[1]
        y_binned =  self.label_encoder.fit_transform(X[self.sector_column])
        self.n_buckets = len(self.label_encoder.classes_)
        # self.bucket_cl: SectorClassifier = self.bucket_cl.set_params(**self.params.get("cls", {})).fit(X, y_binned)
        # groups = self.bucket_cl.predict(X)
        self.bucket_rg = self.bucket_rg.set_params(**self.params.get("rgs", {})).fit(X, y, groups=y_binned)
        # self.bucket_rg = self.bucket_rg.set_params(**self.params.get("rgs", {})).fit(X, y, groups=groups)
        return self

    def predict(self, X, **kwargs) -> ArrayLike:
        # y_binned_pred = self.bucket_cl.predict(X)
        y_binned_pred =  self.label_encoder.transform(X[self.sector_column])
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
        self.label_encoder = self.label_encoder.fit(X_train[self.sector_column])
        y_train_binned = self.label_encoder.transform(X_train[self.sector_column]).reshape(-1, 1)
        y_val_binned = self.label_encoder.transform(X_val[self.sector_column]).reshape(-1, 1)
    
        # best_params_cls, info_cls = self.bucket_cl.optimize(X_train, y_train_binned, X_val, y_val_binned, **kwargs)
        best_params_rgs, info_rgs = self.bucket_rg.optimize(X_train, y_train, X_val, y_val, grp_train=y_train_binned, grp_val=y_val_binned, **kwargs)

        # return {**best_params_cls,**best_params_rgs}, {"classifier":info_cls, "regressor":info_rgs}
        return {"rgs": best_params_rgs}, {"regressor": info_rgs}

    def evaluate(self, y_true, y_pred, **kwargs):
        X_test = kwargs.get("X_test")
        y_true_bins = self.label_encoder.transform(X_test[self.sector_column])

        # y_pred_cl = self.bucket_cl.predict(X_test)
        # results_cl = self.bucket_cl.evaluate(y_true_bins, y_pred_cl)

        y_pred_rg = self.bucket_rg.predict(X_test, groups=y_true_bins)
        results_rg = self.bucket_rg.evaluate(y_true, y_pred_rg)

        results_end_to_end = self._evaluator.evaluate(y_true, y_pred)
        combined_results = {"n_buckets": self.n_buckets, "regressor": results_rg, **results_end_to_end}
        return combined_results