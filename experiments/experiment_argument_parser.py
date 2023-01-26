import argparse


class ExperimentCommandLineParser():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description='Experiment arguments: number of repetitions, what scopes to incorporate (--scope-1 for scope 1 only, --scope-all for all 3 scopes), what file to write to (--append to append to existing file, --new to create new file) and what feature reduction methods to compare (use flag -methods before). Defaults: 10 repititions, --scope-1, --new, all methods.')

        self.parser.add_argument('num_reps', nargs='?', default=2, type=int, help='Number of experiment repititions (default=10)')
        
        self.parser.add_argument('--scope-1', dest='scope', action='store_false', help='(default) use only scope 1')
        self.parser.add_argument('--scope-all', dest='scope', action='store_true', help='use scopes 1, 2, 3')
        self.parser.set_defaults(scope=False)

        self.parser.add_argument('--append', dest='file', action='store_false', help='append results to existing file')
        self.parser.add_argument('--new', dest='file', action='store_true', help='(default) store results in new file')
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
        self.parser.add_argument('-methods', dest='f_r_methods', nargs='*', type=str, default=[DummyFeatureReducer, PCAFeatureSelector, DropFeatureReducer, FeatureAgglomeration, GaussRandProjection, SparseRandProjection, Factor_Analysis], help='Names of feature reduction methods to compare, use flag -methods before specifying methods')
        return super().set_experiment_specific_arguments()