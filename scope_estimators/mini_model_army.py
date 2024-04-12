from base import DefaultRegressorEvaluator, OxariScopeEstimator
from base.helper import BucketScopeDiscretizer, SingleBucketScopeDiscretizer
from base.oxari_types import ArrayLike
from scope_estimators.mma.classifier import (BucketClassifier, UnderfittedBucketClassifier,
                                             BucketClassifierEvauator,
                                             ClassifierOptimizer, MajorityBucketClassifier, RandomGuessBucketClassifier)
from scope_estimators.mma.regressor import (BucketDoubleLevelStackingRegressor, BucketRegressor, BucketStackingRegressor,
                                            EvenWeightBucketRegressor, 
                                            AlternativeCVMetricBucketRegressor,
                                            RegressorOptimizer,
                                            CombinedBucketRegressor, StackingRegressorOptimizer)
from typing_extensions import Self
N_TRIALS = 1
N_STARTUP_TRIALS = 1


class MiniModelArmyEstimator(OxariScopeEstimator):
    def __init__(self, n_buckets=5, cls_optimizer=None, rgs_optimizer=None, bucket_classifier=None, bucket_regressor=None, **kwargs):
        super().__init__(**kwargs)
        self.n_buckets = n_buckets
        self.discretizer = BucketScopeDiscretizer(self.n_buckets)

        self.bucket_cl: BucketClassifier = bucket_classifier or BucketClassifier().set_optimizer(cls_optimizer or ClassifierOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(BucketClassifierEvauator())
        self.bucket_rg: BucketRegressor = bucket_regressor or BucketRegressor().set_optimizer(rgs_optimizer or RegressorOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())
      
    def fit(self, X, y, **kwargs) -> Self:
        # NOTE: Question is whether the linkage between bucket_cl prediction and regression makes sense. I start to believe it does not. 
        # If the classfier predicts one class only the regressor will just use the full data.
        # If the classifier predicts the majority class the model will have one powerful bucket and others are weak.
        # Seperate learning allows to every bucket to learn according to the data distribution. The error does not propagate.  
        self.n_features_in_ = X.shape[1]
        y_binned = self.discretizer.fit_transform(X, y)
        self.bucket_cl: BucketClassifier = self.bucket_cl.set_params(**self.params.get("cls", {})).fit(X, y_binned)
        # groups = self.bucket_cl.predict(X)
        self.bucket_rg = self.bucket_rg.set_params(**self.params.get("rgs", {})).fit(X, y, groups=y_binned.flatten())
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
        self.bucket_cl.set_optimizer(None)
        self.bucket_rg.set_optimizer(None)
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


class EvenWeightMiniModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(n_buckets, cls, rgs, **kwargs)
        self.bucket_rg: BucketRegressor = EvenWeightBucketRegressor().set_optimizer(RegressorOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())

class AlternativeCVMiniModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(n_buckets, cls, rgs, **kwargs)
        self.bucket_rg: BucketRegressor = AlternativeCVMetricBucketRegressor().set_optimizer(RegressorOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())

class CombinedMiniModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(n_buckets, cls, rgs, **kwargs)
        self.bucket_rg: BucketRegressor = CombinedBucketRegressor().set_optimizer(RegressorOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())

class SingleBucketVotingArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, cls={}, rgs={}, **kwargs):
        super().__init__(1, cls, rgs, **kwargs)
        self.discretizer = SingleBucketScopeDiscretizer(n_buckets=1)

class MiniModelArmyClusterBucketEstimator(MiniModelArmyEstimator):
    # TODO: instead of classifier uses clustering for bucketing
    pass

class UnderfittedClsMiniModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(n_buckets, cls, rgs, **kwargs)
        self.bucket_cl: BucketClassifier = UnderfittedBucketClassifier().set_optimizer(ClassifierOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(BucketClassifierEvauator())

class RandomGuessClsMiniModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(n_buckets, cls, rgs, **kwargs)
        self.bucket_cl: BucketClassifier = RandomGuessBucketClassifier().set_optimizer(ClassifierOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(BucketClassifierEvauator())

class MajorityClsMiniModelArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(n_buckets, cls, rgs, **kwargs)
        self.bucket_cl: BucketClassifier = MajorityBucketClassifier().set_optimizer(ClassifierOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(BucketClassifierEvauator())

class BucketStackingArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(n_buckets, cls, rgs, **kwargs)
        self.bucket_rg: BucketRegressor = BucketStackingRegressor().set_optimizer(RegressorOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())

class BucketDoubleLevelStackingArmyEstimator(MiniModelArmyEstimator):
    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(n_buckets, cls, rgs, **kwargs)
        self.bucket_rg: BucketRegressor = BucketDoubleLevelStackingRegressor().set_optimizer(StackingRegressorOptimizer(n_trials=self.n_trials, n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())