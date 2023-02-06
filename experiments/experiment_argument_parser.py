import argparse
from feature_reducers import (DropFeatureReducer, DummyFeatureReducer,
                              FactorAnalysisFeatureReducer, AgglomerateFeatureReducer,
                              GaussRandProjectionFeatureReducer,
                              IsomapDimensionalityFeatureReducer,
                              MDSDimensionalityFeatureReducer, PCAFeatureReducer,
                              SparseRandProjectionFeatureReducer)


class ExperimentCommandLineParser():
    def __init__(self, description) -> None:
        self.parser = argparse.ArgumentParser(description=description)

        self.parser.add_argument('num_reps', nargs='?', default=2, type=int, help='Number of experiment repititions (default=10)')
        
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
        self.parser.add_argument('-methods', dest='f_r_methods', nargs='*', type=str, default=[DummyFeatureReducer, PCAFeatureReducer, DropFeatureReducer, AgglomerateFeatureReducer, GaussRandProjectionFeatureReducer, SparseRandProjectionFeatureReducer, FactorAnalysisFeatureReducer], help='Names of feature reduction methods to compare, use flag -methods before specifying methods')
        return super().set_experiment_specific_arguments()