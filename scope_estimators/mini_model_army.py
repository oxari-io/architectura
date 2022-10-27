
from typing import Union
from base import OxariScopeEstimator
import numpy as np
import pandas as pd

from scope_estimators.mma.classifier import BucketClassifier



class MiniModelArmyEstimator(OxariScopeEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bucket_cl = BucketClassifier(object_filename = "CL_scope1", scope = "scope_1")
        
    def fit(self, X, y, **kwargs) -> "OxariScopeEstimator":
        return super().fit(X, y, **kwargs)
    
    def predict(self, X) -> Union[np.ndarray, pd.DataFrame]:
        return super().predict(X)

    def check_conformance(self):
        pass

    def deploy(self):
        pass

    def run(self):
        pass