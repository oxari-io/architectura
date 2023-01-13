from base import DefaultRegressorEvaluator
from scope_estimators import MiniModelArmyEstimator
from scope_estimators.mma.classifier import BucketClassifier, ClassifierOptimizer, BucketClassifierEvauator
from scope_estimators.mma.regressor import BucketRegressor, RegressorOptimizer
from base.helper import BucketScopeDiscretizer
import xgboost as xgb


class SingleBucketModelRegressorOptimizer(RegressorOptimizer):
    def __init__(self, n_trials=2, n_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__(n_trials, n_startup_trials, sampler, **kwargs)
        self.models = [
            ("XGB", xgb.XGBRegressor),
        ]  

class SingleBucketModelEstimator(MiniModelArmyEstimator):

    def __init__(self, n_buckets=5, cls={}, rgs={}, **kwargs):
        super().__init__(**kwargs)
        self.n_buckets = n_buckets
        self.discretizer = BucketScopeDiscretizer(self.n_buckets)
        self.bucket_cl: BucketClassifier = BucketClassifier().set_optimizer(ClassifierOptimizer(n_trials=self.n_trials,
                                                                                                n_startup_trials=self.n_startup_trials)).set_evaluator(BucketClassifierEvauator())
        self.bucket_rg: BucketRegressor = BucketRegressor().set_optimizer(SingleBucketModelRegressorOptimizer(n_trials=self.n_trials,
                                                                                             n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())
