import argparse

from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
from base.helper import ArcSinhTargetScaler, ArcSinhScaler, DummyFeatureScaler, DummyTargetScaler, LogTargetScaler
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer, FactorAnalysisFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer, PCAFeatureReducer, SparseRandProjectionFeatureReducer)
from base import MAPIEConfidenceEstimator, OxariDataManager, BaselineConfidenceEstimator, JacknifeConfidenceEstimator, DirectLossConfidenceEstimator, PercentileOffsetConfidenceEstimator
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator, UnderfittedClsMiniModelArmyEstimator, PredictMedianEstimator, SingleVoterModelEstimator,
                              SingleBucketVotingArmyEstimator)
from scope_estimators import (BaselineEstimator, MiniModelArmyEstimator, PredictMedianEstimator, EvenWeightMiniModelArmyEstimator)
from scope_estimators import *
import textdistance


class ExperimentCommandLineParser():

    def __init__(self, description='Experiment arguments: number of repetitions, what scopes to incorporate (-s for all 3 scopes), what file to write to (-a to append to existing file) and what scope estimators to compare (write -c before specifying).') -> None:
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

    def _convert_reduction_methods(self, reduction_methods_str_list):
        # if the reduction methods are not strings, they are already in the right format (in that case it was the default argument of parser)
        if not isinstance(reduction_methods_str_list[0], str):
            return reduction_methods_str_list

        config_arg_store = [arg for arg in self.parser._actions if isinstance(arg, argparse._StoreAction) and arg.option_strings.count("-c")][0]
        default_values = config_arg_store.default
        switcher = {cls_obj.__name__:cls_obj for cls_obj in default_values}

        reduction_methods = []
        for method in reduction_methods_str_list:
            m = switcher.get(method)
            if (m != None):
                reduction_methods.append(m)
            else:
                argmin = np.argmin([textdistance.damerau_levenshtein.distance(method, other) for other in switcher.keys()])
                list_of_str_methd_pairs = [l for l in switcher.items()]
                s, c = list_of_str_methd_pairs[argmin]
                print(f"Invalid config: '{method}'. Did you mean '{s}' ({c})?")
                print("Stopping experiment execution.")
                exit()

        return reduction_methods


class FeatureReductionExperimentCommandLineParser(ExperimentCommandLineParser):

    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[
                                     DummyFeatureReducer, PCAFeatureReducer, DropFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer,
                                     SparseRandProjectionFeatureReducer, FactorAnalysisFeatureReducer
                                 ],
                                 dest='configurations',
                                 help='Names of feature reduction methods to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()


class AllComparisonExperimentCommandLineParser(ExperimentCommandLineParser):
    # TODO: Impl: parser for experiment that has multiple configurations
    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[],
                                 dest='configurations',
                                 help='Names of scope estimator methods to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()

class ScopeEstimatorComparisonExperimentCommandLineParser(ExperimentCommandLineParser):

    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[
                                     MiniModelArmyEstimator, GaussianProcessEstimator, XGBEstimator, LGBEstimator, IndependentFeatureVotingRegressionEstimator, KNNEstimator,
                                     BayesianRegressionEstimator, LinearRegressionEstimator, GLMEstimator, LinearSVREstimator, EvenWeightMiniModelArmyEstimator,
                                     SingleVoterModelEstimator, UnderfittedClsMiniModelArmyEstimator, RandomGuessClsMiniModelArmyEstimator, MajorityClsMiniModelArmyEstimator,
                                     MLPEstimator, PLSEstimator, RNEstimator, SingleVoterModelEstimator, SGDEstimator, SupportVectorEstimator, FastSupportVectorEstimator
                                 ],
                                 dest='configurations',
                                 help='Names of scope estimator methods to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()


class BucketingExperimentCommandLineParser(ExperimentCommandLineParser):

    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[SingleBucketVotingArmyEstimator, MiniModelArmyEstimator, BaselineEstimator, PredictMedianEstimator],
                                 dest='configurations',
                                 help='Names of estimators to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()


class VotingVsSingleExperimentCommandLineParser(ExperimentCommandLineParser):

    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[MiniModelArmyEstimator, SingleVoterModelEstimator, BaselineEstimator, PredictMedianEstimator],
                                 dest='configurations',
                                 help='Names of estimators to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()


class WeightedVotingExperimentCommandLineParser(ExperimentCommandLineParser):

    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[BaselineEstimator, MiniModelArmyEstimator, PredictMedianEstimator, EvenWeightMiniModelArmyEstimator, AlternativeCVMiniModelArmyEstimator, CombinedMiniModelArmyEstimator],
                                 dest='configurations',
                                 help='Names of estimators to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()
    
class StackingVsVotingExperimentCommandLineParser(ExperimentCommandLineParser):

    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[BucketStackingArmyEstimator, BucketDoubleLevelStackingArmyEstimator, BaselineEstimator, MiniModelArmyEstimator, EvenWeightMiniModelArmyEstimator],
                                 dest='configurations',
                                 help='Names of estimators to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()


class ClassifierPerformanceExperimentCommandLineParser(ExperimentCommandLineParser):

    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[MiniModelArmyEstimator, UnderfittedClsMiniModelArmyEstimator],
                                 dest='configurations',
                                 help='Names of estimators to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()


class FeatureScalingExperimentCommandLineParser(ExperimentCommandLineParser):

    def set_experiment_specific_arguments(self):
        self.parser.add_argument('-c',
                                 default=[
                                     ArcSinhScaler(),
                                     RobustScaler(),
                                     PowerTransformer(),
                                     StandardScaler(),
                                     DummyFeatureScaler(),
                                     MinMaxScaler(),
                                 ],
                                 dest='configurations',
                                 help='Names of feature scalers to compare',
                                 nargs='*',
                                 type=str)
        self.parser.add_argument('-c2',
                                 default=[
                                     DummyTargetScaler(),
                                     LogTargetScaler(),
                                     ArcSinhTargetScaler(),
                                 ],
                                 dest='configurations',
                                 help='Names of target scalers to compare',
                                 nargs='*',
                                 type=str)
        return super().set_experiment_specific_arguments()