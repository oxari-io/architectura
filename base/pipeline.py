from typing import Union
import sklearn
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
import abc

from base.dataset_loader import OxariDataLoader
from base.estimator import OxariScopeEstimator
from base.imputer import OxariImputer
from base.postprocessor import OxariPostprocessor
from base.preprocessor import OxariPreprocessor
from base import common 

# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
class OxariPipeline(abc.ABC):
    def __init__(
        self,
        dataset: OxariDataLoader = None,
        preprocessor: OxariPreprocessor = None,
        imputer: OxariImputer = None,
        scope_estimator: OxariScopeEstimator = None,
        postprocessor: OxariPostprocessor = None,
        database_deployer=None,
    ):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.imputer = imputer
        self.scope_estimator = scope_estimator
        self.postprocessor = postprocessor
        # self.resources_postprocessor = database_deployer

    @abc.abstractmethod
    def run_pipeline(self, **kwargs):
        # load dataset and hold in class
        #  dataset
        pass
