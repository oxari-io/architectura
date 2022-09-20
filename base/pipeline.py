from typing import Union
import sklearn
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
import abc


# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
class BaseOxariPipeline(abc.ABC):
    def __init__(self, dataset=None, preprocessor=None, missing_value_estimator=None, scope_estimator=None, resources_processor=None, database_deployer=None):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.missing_value_estimator = missing_value_estimator
        self.scope_estimator = scope_estimator
        self.resources_processor = resources_processor
        self.resources_postprocessor = database_deployer
    
    @abc.abstractmethod
    def run_pipeline(**kwargs):
        # load dataset and hold in class
        #  dataset
        pass
    
