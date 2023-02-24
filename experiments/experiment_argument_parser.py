import argparse
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer,
                              FactorAnalysisFeatureReducer, AgglomerateFeatureReducer,
                              GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer, PCAFeatureReducer,
                              SparseRandProjectionFeatureReducer)
from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, JacknifeConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator,
                              PredictMedianEstimator,
                              SingleBucketModelEstimator, SingleBucketVotingArmyEstimator)
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator,
                              PredictMedianEstimator,
                              EvenWeightMiniModelArmyEstimator)

class ExperimentCommandLineParser():
    def __init__(self, description) -> None:
        self.parser = argparse.ArgumentParser(description=description)

        self.parser.add_argument('num_reps', nargs='?', default=10, type=int, help='Number of experiment repititions (default=10)')
        
        self.parser.add_argument('--scope-all', dest='scope', action='store_true', help='use scopes 1, 2, 3')
        self.parser.set_defaults(scope=False)

        self.parser.add_argument('--append', dest='file', action='store_false', help='append results to existing file')
        self.parser.set_defaults(file=True)

        self.set_experiment_specific_arguments()

    def parse_args(self):
        # TODO chekc for illegal formats 
        self.args = self.parser.parse_args()
        return self.args

    def set_experiment_specific_arguments(self):
        pass

class FeatureReductionExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):
        # TODO what if the naming is not exactly a class name, should this be more flexible in accepting names of reduction methods?
        self.parser.add_argument('-methods', dest='configurations', nargs='*', type=str, default=[DummyFeatureReducer, PCAFeatureReducer, DropFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer, SparseRandProjectionFeatureReducer, FactorAnalysisFeatureReducer], help='Names of feature reduction methods to compare, use flag -methods before specifying')
        return super().set_experiment_specific_arguments()

class ConfidenceEstimatorPerformanceExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-config', dest='configurations', nargs='*', type=str, default=[BaselineConfidenceEstimator, JacknifeConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator, MAPIEConfidenceEstimator], help='Names of estimators to compare, use flag -config before specifying')
        return super().set_experiment_specific_arguments()
    
class BucketingExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):

        self.parser.add_argument('-config', dest='configurations', nargs='*', type=str, default=[SingleBucketVotingArmyEstimator, MiniModelArmyEstimator, BaselineEstimator, PredictMedianEstimator], help='Names of estimators to compare, use flag -config before specifying')
        return super().set_experiment_specific_arguments()

class VotingVsSingleExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):

        self.parser.add_argument('-config', dest='configurations', nargs='*', type=str, default=[MiniModelArmyEstimator, SingleBucketModelEstimator, BaselineEstimator, PredictMedianEstimator], help='Names of estimators to compare, use flag -config before specifying')
        return super().set_experiment_specific_arguments()

class WeightedVotingExperimentCommandLineParser(ExperimentCommandLineParser):
    def set_experiment_specific_arguments(self):

        self.parser.add_argument('-config', dest='configurations', nargs='*', type=str, default=[BaselineEstimator, MiniModelArmyEstimator,PredictMedianEstimator, EvenWeightMiniModelArmyEstimator], help='Names of estimators to compare, use flag -config before specifying')
        return super().set_experiment_specific_arguments()
