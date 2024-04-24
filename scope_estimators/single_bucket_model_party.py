import xgboost as xgb

from base import DefaultRegressorEvaluator
from base.helper import BucketScopeDiscretizer
from scope_estimators import MiniModelArmyEstimator
from scope_estimators.mini_model_army import EvenWeightMiniModelArmyEstimator
from scope_estimators.mma.classifier import (BucketClassifier,
                                             BucketClassifierEvauator,
                                             ClassifierOptimizer)
from scope_estimators.mma.regressor import BucketRegressor, RegressorOptimizer

# TODO: Put this into the mma package
class SingleVoterModelRegressorOptimizer(RegressorOptimizer):
    def __init__(self, n_trials=2, n_startup_trials=1, sampler=None, **kwargs) -> None:
        super().__init__(n_trials, n_startup_trials, sampler, **kwargs)
        self.models = [
            ("XGB", xgb.XGBRegressor),
        ]  

class SingleVoterModelEstimator(EvenWeightMiniModelArmyEstimator):

    def __init__(self, n_buckets=10, cls={}, rgs={}, **kwargs):
        super().__init__(**kwargs)
        self.n_buckets = n_buckets
        self.discretizer = BucketScopeDiscretizer(self.n_buckets)
        self.bucket_cl: BucketClassifier = BucketClassifier().set_optimizer(ClassifierOptimizer(n_trials=self.n_trials,
                                                                                                n_startup_trials=self.n_startup_trials)).set_evaluator(BucketClassifierEvauator())
        self.bucket_rg: BucketRegressor = BucketRegressor().set_optimizer(SingleVoterModelRegressorOptimizer(n_trials=self.n_trials,
                                                                                             n_startup_trials=self.n_startup_trials)).set_evaluator(DefaultRegressorEvaluator())
