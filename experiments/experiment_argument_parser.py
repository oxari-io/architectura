import argparse
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer,
                              FactorAnalysisFeatureReducer, AgglomerateFeatureReducer,
                              GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer, PCAFeatureReducer,
                              SparseRandProjectionFeatureReducer)
from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, JacknifeConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator, BadPerformanceMiniModelArmyEstimator, PredictMedianEstimator, SingleBucketModelEstimator, SingleBucketVotingArmyEstimator)
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator,
                              PredictMedianEstimator,
                              EvenWeightMiniModelArmyEstimator)

class ExperimentCommandLineParser():
    def __init__(self, description) -> None:
        self.parser = argparse.ArgumentParser(description=description)

        self.parser.add_argument('num_reps', nargs='?', default=10, type=int, help='Number of experiment repititions (default=10)')
        
        self.parser.add_argument('-s', '--scope-all', dest='scope', action='store_true', help='Use scopes 1, 2, 3')
        self.parser.set_defaults(scope=False)

        self.parser.add_argument('-a', '--append', dest='file', action='store_false', help='Append results to existing file')
        self.parser.set_defaults(file=True)

        self.set_experiment_specific_arguments()

    def parse_args(self):
        self.args = self.parser.parse_args()
        return self.args

    def set_experiment_specific_arguments(self):
        pass

class FeatureReductionExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c', default=[DummyFeatureReducer, PCAFeatureReducer, DropFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer, SparseRandProjectionFeatureReducer, FactorAnalysisFeatureReducer], dest='configurations', help='Names of feature reduction methods to compare', nargs='*', type=str)
        return super().set_experiment_specific_arguments()
    
class BucketingExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c', default=[SingleBucketVotingArmyEstimator, MiniModelArmyEstimator, BaselineEstimator, PredictMedianEstimator], dest='configurations', help='Names of estimators to compare', nargs='*', type=str)
        return super().set_experiment_specific_arguments()

class VotingVsSingleExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c', default=[MiniModelArmyEstimator, SingleBucketModelEstimator, BaselineEstimator, PredictMedianEstimator], dest='configurations', help='Names of estimators to compare', nargs='*', type=str)
        return super().set_experiment_specific_arguments()

class WeightedVotingExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c', default=[BaselineEstimator, MiniModelArmyEstimator, PredictMedianEstimator, EvenWeightMiniModelArmyEstimator], dest='configurations', help='Names of estimators to compare', nargs='*', type=str)
        return super().set_experiment_specific_arguments()

class ClassifierPerformanceExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c', default=[MiniModelArmyEstimator, BadPerformanceMiniModelArmyEstimator], dest='configurations', help='Names of estimators to compare', nargs='*', type=str)
        return super().set_experiment_specific_arguments()